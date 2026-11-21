"""Explicit availability for registered actions.

An action can be registered but unavailable on the current page because its
domain filter does not match, because it has an empty domain list, because the
page URL is unknown, or because the action was excluded at registration time.
This module gives those states one typed vocabulary for execution, errors, and
status.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import urlsplit

MAX_ACTION_NAMES = 50
MAX_FILTERED_ACTIONS = 50
MAX_EXCLUDED_ACTIONS = 50
MAX_DOMAINS = 8
MAX_ACTION_NAME_CHARS = 80
MAX_DOMAIN_CHARS = 80
MAX_PAGE_ORIGIN_CHARS = 120
MAX_ERROR_CHARS = 800
ERROR_AVAILABLE_SAMPLE = 10
logger = logging.getLogger(__name__)


class ActionAvailabilityCode(StrEnum):
    """Why an action is or is not available on one page."""

    ACTION_UNKNOWN = "action_unknown"
    ACTION_EXCLUDED = "action_excluded"
    DOMAIN_FILTERED = "domain_filtered"
    EMPTY_DOMAINS = "empty_domains"
    PAGE_UNKNOWN = "page_unknown"


class ToolAvailabilityPhase(StrEnum):
    """The aggregate availability state for one registry and page."""

    NOT_INITIALIZED = "not_initialized"
    READY = "ready"
    PAGE_FILTERED = "page_filtered"
    PAGE_UNKNOWN = "page_unknown"


def normalize_page_url(url: str | None) -> str | None:
    """Return a usable HTTP(S) page URL, or None for an unknown page context.

    A falsey URL is not a wildcard. ``about:blank``, ``chrome://newtab``, data
    URLs, and malformed values are not page contexts for domain-filtered
    actions.
    """
    if not isinstance(url, str):
        return None
    value = url.strip()
    if not value or "\\" in value or any(character.isspace() for character in value):
        return None
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        _port = parsed.port
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not host:
        return None
    return value


def page_origin(url: str | None) -> str | None:
    """Return a bounded, origin-only page URL for diagnostics.

    Path, query, fragment, and userinfo are deliberately discarded. Domain
    matching only needs the scheme, host, and port.
    """
    normalized = normalize_page_url(url)
    if normalized is None:
        return None
    try:
        parsed = urlsplit(normalized)
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return None
    if not host:
        return None
    scheme = parsed.scheme.lower()
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    origin = f"{scheme}://{host}"
    if port is not None:
        origin = f"{origin}:{port}"
    return _clip(origin, MAX_PAGE_ORIGIN_CHARS)


def active_page_url_from_snapshot(browser: object) -> str | None:
    """Return the active tab URL from a browser lifecycle snapshot."""
    if not isinstance(browser, dict):
        return None
    active_target_id = browser.get("active_target_id")
    tabs = browser.get("tabs")
    if isinstance(tabs, list):
        for tab in tabs:
            if not isinstance(tab, dict):
                continue
            if tab.get("target_id") == active_target_id or tab.get("active") is True:
                url = tab.get("url")
                if isinstance(url, str) and url:
                    return url
    url = browser.get("url")
    return url if isinstance(url, str) and url else None


def tool_availability_payload(tools: object | None, browser: object) -> dict[str, object]:
    """Return a JSON-safe tools status payload, or a safe not-initialized view."""
    if tools is None:
        return ToolAvailabilitySnapshot.not_initialized().to_dict()
    reader = getattr(tools, "availability_snapshot", None)
    if callable(reader):
        try:
            snapshot = reader(active_page_url_from_snapshot(browser))
        except Exception:
            logger.warning("Tool availability snapshot failed", exc_info=True)
            snapshot = None
        if isinstance(snapshot, ToolAvailabilitySnapshot):
            return snapshot.to_dict()
    return ToolAvailabilitySnapshot.not_initialized().to_dict()


def _clip(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    if limit <= 3:
        return value[:limit]
    return f"{value[: limit - 3]}..."


def _bounded_names(values: Iterable[str], limit: int = MAX_ACTION_NAMES) -> tuple[tuple[str, ...], bool]:
    raw = [str(value) for value in values if value is not None]
    cleaned = sorted({_clip(value, MAX_ACTION_NAME_CHARS) for value in raw})
    truncated = len(set(raw)) > limit or any(len(value) > MAX_ACTION_NAME_CHARS for value in raw)
    return tuple(cleaned[:limit]), truncated


def _bounded_domains(values: Iterable[str]) -> tuple[tuple[str, ...], bool]:
    raw = [str(value) for value in values if value is not None]
    cleaned = sorted({_clip(value, MAX_DOMAIN_CHARS) for value in raw})
    truncated = len(set(raw)) > MAX_DOMAINS or any(len(value) > MAX_DOMAIN_CHARS for value in raw)
    return tuple(cleaned[:MAX_DOMAINS]), truncated


@dataclass(frozen=True)
class FilteredAction:
    """One registered action that cannot run on the observed page."""

    name: str
    domains: tuple[str, ...]
    reason: ActionAvailabilityCode

    def to_dict(self) -> dict[str, object]:
        domains, _ = _bounded_domains(self.domains)
        return {
            "name": _clip(self.name, MAX_ACTION_NAME_CHARS),
            "domains": list(domains),
            "reason": self.reason.value,
        }


@dataclass(frozen=True)
class ToolAvailabilitySnapshot:
    """A read-only view of action availability for one page."""

    phase: ToolAvailabilityPhase
    page_url: str | None
    registered: tuple[str, ...]
    available: tuple[str, ...]
    filtered: tuple[FilteredAction, ...]
    excluded: tuple[str, ...]
    truncated: bool = False

    @classmethod
    def not_initialized(cls) -> ToolAvailabilitySnapshot:
        return cls(
            phase=ToolAvailabilityPhase.NOT_INITIALIZED,
            page_url=None,
            registered=(),
            available=(),
            filtered=(),
            excluded=(),
        )

    def to_dict(self) -> dict[str, object]:
        available, available_truncated = _bounded_names(self.available)
        excluded, excluded_truncated = _bounded_names(self.excluded, MAX_EXCLUDED_ACTIONS)
        filtered = self.filtered[:MAX_FILTERED_ACTIONS]
        filtered_truncated = len(self.filtered) > MAX_FILTERED_ACTIONS
        filtered_fields_truncated = any(
            len(item.name) > MAX_ACTION_NAME_CHARS
            or len(item.domains) > MAX_DOMAINS
            or any(len(domain) > MAX_DOMAIN_CHARS for domain in item.domains)
            for item in self.filtered
        )
        filtered_payload = [item.to_dict() for item in filtered]
        origin = page_origin(self.page_url) if self.page_url else None
        origin_truncated = bool(origin and origin.endswith("..."))
        return {
            "phase": self.phase.value,
            "page_url": origin,
            "registered_count": len(self.registered),
            "available_count": len(self.available),
            "filtered_count": len(self.filtered),
            "excluded_count": len(self.excluded),
            "available": list(available),
            "filtered": filtered_payload,
            "excluded": list(excluded),
            "truncated": (
                self.truncated
                or available_truncated
                or excluded_truncated
                or filtered_truncated
                or filtered_fields_truncated
                or origin_truncated
            ),
        }


def _default_hint(code: ActionAvailabilityCode) -> str:
    if code is ActionAvailabilityCode.ACTION_EXCLUDED:
        return "This action is disabled by configuration. Use a registered action or change exclude_actions."
    if code is ActionAvailabilityCode.ACTION_UNKNOWN:
        return "Use a registered action. Run get_status and read tools.available."
    if code is ActionAvailabilityCode.EMPTY_DOMAINS:
        return "Fix the action registration; an empty domain list matches no page."
    if code is ActionAvailabilityCode.PAGE_UNKNOWN:
        return (
            "Read browser state with `await browser.get_browser_state_summary()` before calling a page-filtered action."
        )
    return "Use an action available on this page or switch to a matching page. Refresh browser state before retrying."


class ActionAvailabilityError(ValueError):
    """A typed action-availability failure with a concrete next step."""

    def __init__(
        self,
        code: ActionAvailabilityCode,
        *,
        action: str,
        display_name: str | None = None,
        page_url: str | None = None,
        domains: Iterable[str] = (),
        available: Iterable[str] = (),
        hint: str | None = None,
    ):
        self.code = code
        self.action = action
        raw_display_name = display_name or action
        raw_hint = hint or _default_hint(code)
        self.display_name = _clip(raw_display_name, MAX_ACTION_NAME_CHARS)
        self.page_url = page_origin(page_url)
        self.domains, domains_truncated = _bounded_domains(domains)
        self.available, available_truncated = _bounded_names(available, ERROR_AVAILABLE_SAMPLE)
        self.hint = _clip(raw_hint, 300)
        self.truncated = (
            domains_truncated
            or available_truncated
            or len(raw_display_name) > MAX_ACTION_NAME_CHARS
            or len(raw_hint) > 300
            or bool(self.page_url and self.page_url.endswith("..."))
        )
        message = self._format()
        if len(message) > MAX_ERROR_CHARS:
            message = _clip(message, MAX_ERROR_CHARS)
            self.truncated = True
        self.message = message
        super().__init__(message)

    def _format(self) -> str:
        page = f"on {self.page_url}" if self.page_url else "without a known page URL"
        if self.code is ActionAvailabilityCode.ACTION_EXCLUDED:
            reason = "is disabled by configuration."
        elif self.code is ActionAvailabilityCode.ACTION_UNKNOWN:
            reason = "is not registered."
        elif self.code is ActionAvailabilityCode.EMPTY_DOMAINS:
            reason = "has an empty domain list and cannot run on any page."
        elif self.code is ActionAvailabilityCode.PAGE_UNKNOWN:
            reason = "requires a known HTTP(S) page URL."
        else:
            reason = "is not allowed on this page."
        parts = [f"Action availability [{self.code.value}]: {self.display_name}() {reason} {page}."]
        if self.domains:
            parts.append(f"Allowed domains: {', '.join(self.domains)}.")
        if self.available:
            parts.append(f"Available now: {', '.join(self.available)}.")
        body = " ".join(parts)
        max_body = max(0, MAX_ERROR_CHARS - len(self.hint) - 1)
        if len(body) > max_body:
            body = _clip(body, max_body)
            self.truncated = True
        return f"{body} {self.hint}".strip()

    def with_action(self, action: str, *, display_name: str | None = None) -> ActionAvailabilityError:
        """Return the same failure named by the public callable."""
        return ActionAvailabilityError(
            self.code,
            action=action,
            display_name=display_name or self.display_name,
            page_url=self.page_url,
            domains=self.domains,
            available=self.available,
            hint=self.hint,
        )
