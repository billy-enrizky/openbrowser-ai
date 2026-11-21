"""Behavior tests for explicit tool-availability state."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from openbrowser.code_use.namespace import create_namespace
from openbrowser.models import ActionResult
from openbrowser.tools.action_availability import (
    ActionAvailabilityCode,
    ActionAvailabilityError,
    ToolAvailabilityPhase,
    ToolAvailabilitySnapshot,
    active_page_url_from_snapshot,
)
from openbrowser.tools.registry.service import Registry
from openbrowser.tools.service import Tools


def _make_registry(exclude_actions=None):
    registry = Registry(exclude_actions=exclude_actions)
    calls: list[str] = []

    @registry.action("Unfiltered action")
    async def open_action():
        calls.append("open")
        return "open"

    @registry.action("Google action", domains=["*.google.com"])
    async def google_action():
        calls.append("google")
        return "google"

    return registry, calls


def _session_on(page_url: str):
    session = MagicMock()
    session.get_current_page_url = AsyncMock(return_value=page_url)
    session.browser_profile = MagicMock()
    session.browser_profile.downloads_path = None
    return session


def test_snapshot_distinguishes_available_filtered_and_unknown_pages():
    registry, _ = _make_registry()

    matching = registry.availability_snapshot("https://www.google.com/search")
    assert matching.phase is ToolAvailabilityPhase.READY
    assert matching.available == ("google_action", "open_action")
    assert matching.filtered == ()

    nonmatching = registry.availability_snapshot("https://example.com")
    assert nonmatching.phase is ToolAvailabilityPhase.PAGE_FILTERED
    assert nonmatching.available == ("open_action",)
    assert len(nonmatching.filtered) == 1
    assert nonmatching.filtered[0].reason is ActionAvailabilityCode.DOMAIN_FILTERED

    unknown = registry.availability_snapshot(None)
    assert unknown.phase is ToolAvailabilityPhase.PAGE_UNKNOWN
    assert unknown.filtered[0].reason is ActionAvailabilityCode.PAGE_UNKNOWN


def test_initialized_registry_with_only_unfiltered_actions_is_ready_without_a_page():
    registry = Registry()

    @registry.action("General")
    async def general():
        return "ok"

    snapshot = registry.availability_snapshot(None)

    assert snapshot.phase is ToolAvailabilityPhase.READY
    assert snapshot.available == ("general",)
    assert snapshot.filtered == ()


@pytest.mark.parametrize(
    "page_url",
    ["", "about:blank", "chrome://newtab", "data:text/html,hello", "file:///tmp/report.pdf"],
)
def test_non_http_or_blank_pages_are_unknown_for_filtered_actions(page_url):
    registry, _ = _make_registry()

    with pytest.raises(ActionAvailabilityError) as exc_info:
        registry.check_action_available("google_action", page_url=page_url)

    assert exc_info.value.code is ActionAvailabilityCode.PAGE_UNKNOWN


def test_empty_domain_list_is_deny_all_and_takes_precedence_over_page_unknown():
    registry = Registry()

    @registry.action("No domains", domains=[])
    async def no_domains():
        return "never"

    snapshot = registry.availability_snapshot(None)
    assert snapshot.phase is ToolAvailabilityPhase.PAGE_UNKNOWN
    assert snapshot.filtered[0].reason is ActionAvailabilityCode.EMPTY_DOMAINS

    with pytest.raises(ActionAvailabilityError) as exc_info:
        registry.check_action_available("no_domains", page_url=None)
    assert exc_info.value.code is ActionAvailabilityCode.EMPTY_DOMAINS

    with pytest.raises(ActionAvailabilityError) as exc_info:
        registry.check_action_available("no_domains", page_url="https://example.com")
    assert exc_info.value.code is ActionAvailabilityCode.EMPTY_DOMAINS


def test_excluded_and_unknown_actions_have_distinct_codes():
    registry = Registry(exclude_actions=["blocked"])

    @registry.action("Blocked")
    async def blocked():
        return "blocked"

    with pytest.raises(ActionAvailabilityError) as excluded:
        registry.check_action_available("blocked")
    assert excluded.value.code is ActionAvailabilityCode.ACTION_EXCLUDED

    with pytest.raises(ActionAvailabilityError) as unknown:
        registry.check_action_available("missing")
    assert unknown.value.code is ActionAvailabilityCode.ACTION_UNKNOWN


@pytest.mark.asyncio
async def test_execute_action_gates_before_parameter_validation_and_execution():
    registry, calls = _make_registry()

    with pytest.raises(ActionAvailabilityError) as exc_info:
        await registry.execute_action(
            "google_action",
            {"unexpected": "value"},
            page_url="https://example.com",
        )

    assert exc_info.value.code is ActionAvailabilityCode.DOMAIN_FILTERED
    assert calls == []

    result = await registry.execute_action(
        "google_action",
        {},
        page_url="https://www.google.com",
    )
    assert result == "google"
    assert calls == ["google"]


@pytest.mark.asyncio
async def test_execute_action_fails_closed_without_a_page_context():
    registry, calls = _make_registry()

    with pytest.raises(ActionAvailabilityError) as exc_info:
        await registry.execute_action("google_action", {})

    assert exc_info.value.code is ActionAvailabilityCode.PAGE_UNKNOWN
    assert calls == []


@pytest.mark.asyncio
async def test_live_browser_session_is_authoritative_over_a_supplied_page_url():
    registry, calls = _make_registry()
    session = _session_on("https://example.com")

    with pytest.raises(ActionAvailabilityError) as exc_info:
        await registry.execute_action(
            "google_action",
            {},
            browser_session=session,
            page_url="https://www.google.com",
        )

    assert exc_info.value.code is ActionAvailabilityCode.DOMAIN_FILTERED
    assert calls == []


@pytest.mark.asyncio
async def test_execute_action_forwards_page_url_to_the_special_parameter():
    registry = Registry()

    @registry.action("Needs page", domains=["*.example.com"])
    async def needs_page(page_url: str):
        return page_url

    result = await registry.execute_action(
        "needs_page",
        {},
        page_url="https://example.com/page?q=1",
    )

    assert result == "https://example.com/page?q=1"


@pytest.mark.asyncio
async def test_direct_decorated_call_is_gated():
    registry = Registry()
    calls: list[str] = []

    @registry.action("Google action", domains=["*.google.com"])
    async def google_action():
        calls.append("google")
        return "google"

    with pytest.raises(ActionAvailabilityError) as exc_info:
        await google_action(page_url="https://example.com")
    assert exc_info.value.code is ActionAvailabilityCode.DOMAIN_FILTERED
    assert calls == []

    result = await google_action(page_url="https://www.google.com")
    assert result == "google"
    assert calls == ["google"]


@pytest.mark.asyncio
async def test_excluded_decorated_action_never_runs_its_body():
    registry = Registry(exclude_actions=["blocked"])
    calls: list[str] = []

    @registry.action("Blocked")
    async def blocked():
        calls.append("ran")
        return "blocked"

    with pytest.raises(ActionAvailabilityError) as exc_info:
        await blocked()

    assert exc_info.value.code is ActionAvailabilityCode.ACTION_EXCLUDED
    assert calls == []


@pytest.mark.asyncio
async def test_tools_act_preserves_the_availability_code():
    tools = Tools()
    calls: list[str] = []

    @tools.registry.action("Google action", domains=["*.google.com"])
    async def google_action():
        calls.append("ran")
        return "google"

    ActionModelClass = tools.registry.create_action_model(page_url="https://www.google.com")
    action = ActionModelClass.model_validate({"google_action": {}})
    session = _session_on("https://example.com")

    result = await tools.act(action, browser_session=session)

    assert isinstance(result, ActionResult)
    assert result.error_code == ActionAvailabilityCode.DOMAIN_FILTERED.value
    assert calls == []


@pytest.mark.asyncio
async def test_tools_direct_action_api_gates_before_parameter_validation():
    tools = Tools()
    calls: list[str] = []

    @tools.registry.action("Google action", domains=["*.google.com"])
    async def google_action(text: str):
        calls.append(text)
        return text

    with pytest.raises(ActionAvailabilityError) as exc_info:
        await tools.google_action(  # type: ignore[attr-defined]
            browser_session=_session_on("https://example.com")
        )

    assert exc_info.value.code is ActionAvailabilityCode.DOMAIN_FILTERED
    assert calls == []


@pytest.mark.asyncio
async def test_namespace_input_text_alias_uses_the_registry_key():
    tools = Tools()

    @tools.registry.action("Custom input")
    async def input():
        return "ok"

    namespace = create_namespace(_session_on("https://example.com"), tools=tools)
    result = await namespace["input_text"]()

    assert result == "ok"


@pytest.mark.asyncio
async def test_namespace_refuses_filtered_action_before_parameter_validation():
    tools = Tools()
    calls: list[str] = []

    @tools.registry.action("Google action", domains=["*.google.com"])
    async def google_action(text: str):
        calls.append(text)
        return text

    namespace = create_namespace(_session_on("https://example.com"), tools=tools)

    with pytest.raises(ActionAvailabilityError) as exc_info:
        await namespace["google_action"]()

    assert exc_info.value.code is ActionAvailabilityCode.DOMAIN_FILTERED
    assert calls == []


@pytest.mark.asyncio
async def test_namespace_unfiltered_action_still_runs():
    tools = Tools()

    @tools.registry.action("Open action")
    async def open_action():
        return "ok"

    namespace = create_namespace(_session_on("https://example.com"), tools=tools)

    assert await namespace["open_action"]() == "ok"


def test_availability_error_is_bounded_and_origin_only():
    error = ActionAvailabilityError(
        ActionAvailabilityCode.DOMAIN_FILTERED,
        action="x" * 300,
        page_url="https://user:pass@example.com:8443/private/report.pdf?token=secret#frag",
        domains=[f"{index}.example.com" for index in range(20)],
        available=[f"action_{index}" for index in range(20)],
    )
    text = str(error)

    assert len(text) <= 800
    assert "secret" not in text
    assert "user" not in text
    assert "/private" not in text
    assert "#frag" not in text
    assert "https://example.com:8443" in text
    assert "Use an action available" in text
    assert error.truncated is True


def test_snapshot_overflow_is_deterministic_and_reports_total_counts():
    registry = Registry()
    for index in range(60):

        def make_action(position: int):
            async def action():
                return position

            action.__name__ = f"action_{position:02d}"
            return action

        registry.action("Overflow")(make_action(index))

    snapshot = registry.availability_snapshot(
        "https://example.com/" + "x" * 200,
    ).to_dict()

    assert snapshot["registered_count"] == 60
    assert snapshot["available_count"] == 60
    assert len(snapshot["available"]) == 50
    assert snapshot["truncated"] is True
    assert snapshot["page_url"] == "https://example.com"


def test_snapshot_redacts_page_url_to_origin():
    registry, _ = _make_registry()
    snapshot = registry.availability_snapshot(
        "https://user:pass@example.com:8443/private/page?token=secret#frag"
    ).to_dict()

    assert snapshot["page_url"] == "https://example.com:8443"
    assert "secret" not in str(snapshot)
    assert "/private" not in str(snapshot)


def test_active_page_url_from_snapshot_uses_the_active_tab():
    browser = {
        "active_target_id": "tab-2",
        "tabs": [
            {"target_id": "tab-1", "url": "https://first.example"},
            {"target_id": "tab-2", "url": "https://second.example/page"},
        ],
    }

    assert active_page_url_from_snapshot(browser) == "https://second.example/page"


def test_not_initialized_snapshot_shape():
    assert ToolAvailabilitySnapshot.not_initialized().to_dict() == {
        "phase": "not_initialized",
        "page_url": None,
        "registered_count": 0,
        "available_count": 0,
        "filtered_count": 0,
        "excluded_count": 0,
        "available": [],
        "filtered": [],
        "excluded": [],
        "truncated": False,
    }


def test_long_origin_and_nested_domain_overflow_set_truncated():
    registry = Registry()

    @registry.action(
        "Many domains",
        domains=[f"{index}." + "a" * 100 + ".example.com" for index in range(12)],
    )
    async def many_domains():
        return "never"

    snapshot = registry.availability_snapshot("https://" + "b" * 200 + ".example.com/path").to_dict()

    assert snapshot["page_url"].endswith("...")
    assert snapshot["truncated"] is True
    assert len(snapshot["filtered"][0]["domains"]) == 8

    error = ActionAvailabilityError(
        ActionAvailabilityCode.DOMAIN_FILTERED,
        action="many_domains",
        page_url="https://" + "b" * 200 + ".example.com/path",
        domains=[f"{index}." + "a" * 100 + ".example.com" for index in range(12)],
    )
    assert error.truncated is True
    assert error.page_url is not None and error.page_url.endswith("...")


def test_single_long_value_sets_truncated_and_keeps_the_hint():
    registry = Registry()
    long_name = "action_" + "x" * 200
    long_domain = "a" * 200 + ".example.com"

    @registry.action("Long", domains=[long_domain])
    async def long_action():
        return "never"

    # Rename the registry key to an overlong value without exceeding list limits.
    action = registry.registry.actions.pop("long_action")
    action.name = long_name
    registry.registry.actions[long_name] = action

    snapshot = registry.availability_snapshot("https://example.com").to_dict()
    assert snapshot["truncated"] is True
    assert len(snapshot["filtered"][0]["name"]) <= 80
    assert len(snapshot["filtered"][0]["domains"][0]) <= 80

    error = ActionAvailabilityError(
        ActionAvailabilityCode.DOMAIN_FILTERED,
        action=long_name,
        display_name="d" * 200,
        page_url="https://example.com",
        domains=[long_domain],
    )
    assert error.truncated is True
    assert len(error.display_name) <= 80
    assert "Use an action available" in str(error)


def test_invalid_port_is_page_unknown():
    registry, _ = _make_registry()

    with pytest.raises(ActionAvailabilityError) as exc_info:
        registry.check_action_available(
            "google_action",
            page_url="https://example.com:70000",
        )

    assert exc_info.value.code is ActionAvailabilityCode.PAGE_UNKNOWN


def test_non_http_page_is_not_advertised_or_available():
    registry = Registry()

    @registry.action("Extension", domains=["chrome-extension://*"])
    async def extension_action():
        return "never"

    model = registry.create_action_model(page_url="chrome-extension://abc")
    with pytest.raises(ValidationError):
        model.model_validate({"extension_action": {}})

    assert "extension_action" not in registry.get_prompt_description(page_url="chrome-extension://abc")


def test_empty_domain_action_is_omitted_from_model_and_prompt():
    registry = Registry()

    @registry.action("General")
    async def general():
        return "ok"

    @registry.action("Empty", domains=[])
    async def empty_domain():
        return "never"

    model = registry.create_action_model(page_url="https://example.com")
    with pytest.raises(ValidationError):
        model.model_validate({"empty_domain": {}})
    model.model_validate({"general": {}})

    model_without_page = registry.create_action_model(page_url=None)
    with pytest.raises(ValidationError):
        model_without_page.model_validate({"empty_domain": {}})
    model_without_page.model_validate({"general": {}})

    assert "empty_domain" not in registry.get_prompt_description(page_url="https://example.com")
    assert "empty_domain" not in registry.get_prompt_description(page_url=None)


@pytest.mark.asyncio
async def test_direct_call_forwards_the_live_session_page_url():
    registry = Registry()

    @registry.action("Needs page")
    async def needs_page(page_url: str):
        return page_url

    session = _session_on("https://live.example.com/page")

    result = await needs_page(
        browser_session=session,
        page_url="https://supplied.example.com/page",
    )

    assert result == "https://live.example.com/page"


@pytest.mark.asyncio
async def test_padded_supplied_url_is_normalized_before_dispatch():
    registry = Registry()

    @registry.action("Needs page")
    async def needs_page(page_url: str):
        return page_url

    result = await needs_page(page_url="  https://example.com/page  ")

    assert result == "https://example.com/page"


@pytest.mark.asyncio
async def test_action_page_url_default_is_preserved_when_none_is_available():
    registry = Registry()

    @registry.action("Fallback")
    async def fallback(page_url: str = "fallback"):
        return page_url

    result = await registry.execute_action("fallback", {})

    assert result == "fallback"


@pytest.mark.asyncio
async def test_namespace_preserves_action_page_url_default():
    tools = Tools()

    @tools.registry.action("Fallback")
    async def fallback(page_url: str = "fallback"):
        return page_url

    namespace = create_namespace(_session_on("https://example.com"), tools=tools)

    assert await namespace["fallback"]() == "fallback"


@pytest.mark.asyncio
async def test_unreadable_live_url_does_not_fall_back_to_the_supplied_url():
    registry, calls = _make_registry()
    session = MagicMock()
    session.get_current_page_url = AsyncMock(side_effect=RuntimeError("live URL failed"))

    with pytest.raises(ActionAvailabilityError) as exc_info:
        await registry.execute_action(
            "google_action",
            {},
            browser_session=session,
            page_url="https://www.google.com",
        )

    assert exc_info.value.code is ActionAvailabilityCode.PAGE_UNKNOWN
    assert calls == []


@pytest.mark.asyncio
async def test_empty_domain_dispatch_fails_before_the_body_without_a_page():
    registry = Registry()
    calls: list[str] = []

    @registry.action("Empty", domains=[])
    async def empty_domain():
        calls.append("ran")
        return "never"

    with pytest.raises(ActionAvailabilityError) as exc_info:
        await registry.execute_action("empty_domain", {})

    assert exc_info.value.code is ActionAvailabilityCode.EMPTY_DOMAINS
    assert calls == []


def test_error_display_name_truncation_sets_the_flag():
    error = ActionAvailabilityError(
        ActionAvailabilityCode.DOMAIN_FILTERED,
        action="short",
        display_name="d" * 200,
        domains=["example.com"],
    )

    assert error.truncated is True
    assert len(error.display_name) <= 80


def test_error_hint_truncation_sets_the_flag():
    error = ActionAvailabilityError(
        ActionAvailabilityCode.DOMAIN_FILTERED,
        action="short",
        hint="h" * 400,
        domains=["example.com"],
    )

    assert error.truncated is True
    assert len(error.hint) <= 300


@pytest.mark.asyncio
async def test_excluded_evaluate_is_not_available_in_the_namespace():
    tools = Tools(exclude_actions=["evaluate"])
    namespace = create_namespace(_session_on("https://example.com"), tools=tools)

    with pytest.raises(ActionAvailabilityError) as exc_info:
        await namespace["evaluate"]("document.title")

    assert exc_info.value.code is ActionAvailabilityCode.ACTION_EXCLUDED


@pytest.mark.asyncio
async def test_excluded_tools_attribute_raises_typed_error():
    tools = Tools(exclude_actions=["screenshot"])

    with pytest.raises(ActionAvailabilityError) as exc_info:
        await tools.screenshot(browser_session=_session_on("https://example.com"))

    assert exc_info.value.code is ActionAvailabilityCode.ACTION_EXCLUDED


def test_invalid_page_prompt_does_not_inject_unfiltered_actions():
    registry, _ = _make_registry()

    assert registry.get_prompt_description(page_url="about:blank") == ""
    assert "open_action" in registry.get_prompt_description(page_url=None)


@pytest.mark.asyncio
async def test_backslash_url_is_page_unknown():
    registry, _ = _make_registry()

    with pytest.raises(ActionAvailabilityError) as exc_info:
        registry.check_action_available(
            "google_action",
            page_url="https://evil.com\\@good.com",
        )

    assert exc_info.value.code is ActionAvailabilityCode.PAGE_UNKNOWN


@pytest.mark.asyncio
async def test_direct_call_preserves_default_when_supplied_url_is_invalid():
    registry = Registry()

    @registry.action("Fallback")
    async def fallback(page_url: str = "fallback"):
        return page_url

    assert await fallback(page_url="about:blank") == "fallback"


def test_action_result_error_code_is_typed():
    result = ActionResult(error_code="domain_filtered")

    assert result.error_code is ActionAvailabilityCode.DOMAIN_FILTERED
