"""Behavior checks for deferred construction of browser event models."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from pydantic.errors import PydanticInvalidForJsonSchema

ROOT = Path(__file__).resolve().parents[1]

EVENT_CLASSES = (
    "AboutBlankDVDScreensaverShownEvent",
    "AgentFocusChangedEvent",
    "BrowserConnectedEvent",
    "BrowserErrorEvent",
    "BrowserKillEvent",
    "BrowserLaunchEvent",
    "BrowserStartEvent",
    "BrowserStateRequestEvent",
    "BrowserStopEvent",
    "BrowserStoppedEvent",
    "ClickElementEvent",
    "CloseTabEvent",
    "DialogOpenedEvent",
    "ElementSelectedEvent",
    "FileDownloadedEvent",
    "GetDropdownOptionsEvent",
    "GoBackEvent",
    "GoForwardEvent",
    "LoadStorageStateEvent",
    "NavigateToUrlEvent",
    "NavigationCompleteEvent",
    "NavigationFinishedEvent",
    "NavigationStartedEvent",
    "RefreshEvent",
    "SaveStorageStateEvent",
    "ScreenshotEvent",
    "ScrollEvent",
    "ScrollToTextEvent",
    "SelectDropdownOptionEvent",
    "SendKeysEvent",
    "StorageStateFailedEvent",
    "StorageStateLoadedEvent",
    "StorageStateSavedEvent",
    "SwitchTabEvent",
    "TabClosedEvent",
    "TabClosingEvent",
    "TabCreatedEvent",
    "TargetCrashedEvent",
    "TypeTextEvent",
    "UploadFileEvent",
    "WaitEvent",
)

RESULT_TYPES = {
    "BrowserLaunchEvent": "openbrowser.browser.events.BrowserLaunchResult",
    "BrowserStateRequestEvent": "openbrowser.browser.views.BrowserStateSummary",
    "GoBackEvent": "builtins.str",
    "GoForwardEvent": "builtins.str",
    "RefreshEvent": "builtins.str",
    "ScreenshotEvent": "builtins.str",
    "SwitchTabEvent": "builtins.str",
}

REBUILT_BY_TOOLS = {
    "ClickElementEvent",
    "ScrollEvent",
    "TypeTextEvent",
    "UploadFileEvent",
}
REBUILT_BY_DEFAULT_ACTION = {
    "GetDropdownOptionsEvent",
    "SelectDropdownOptionEvent",
}

_PROBE = r"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))

import openbrowser.browser.events as events
from bubus import BaseEvent
from bubus.models import _extract_basemodel_generic_arg


def type_name(value):
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    return repr(value)


def event_classes():
    return sorted(
        (name, value)
        for name, value in vars(events).items()
        if isinstance(value, type)
        and issubclass(value, BaseEvent)
        and value is not BaseEvent
        and value.__module__ == events.__name__
        and "[" not in value.__name__
    )


classes = event_classes()
result_types = {}
for name, value in classes:
    try:
        result_types[name] = type_name(_extract_basemodel_generic_arg(value))
    except Exception as error:
        result_types[name] = f"{type(error).__name__}: {error}"

payload = {
    "classes": [name for name, _ in classes],
    "defer_build": {
        name: bool(getattr(value, "model_config", {}).get("defer_build", False))
        for name, value in classes
    },
    "complete_after_events": {
        name: bool(getattr(value, "__pydantic_complete__", False))
        for name, value in classes
    },
    "result_types": result_types,
    "session_module_loaded": "openbrowser.browser.session" in sys.modules,
    "browser_launch_result": {
        "is_event": issubclass(events.BrowserLaunchResult, BaseEvent),
        "defer_build": bool(
            getattr(events.BrowserLaunchResult, "model_config", {}).get("defer_build", False)
        ),
    },
}

import openbrowser.tools.service

payload["complete_after_tools"] = {
    name: bool(getattr(value, "__pydantic_complete__", False))
    for name, value in classes
}

import openbrowser.browser.watchdogs.default_action_watchdog

payload["complete_after_default_action"] = {
    name: bool(getattr(value, "__pydantic_complete__", False))
    for name, value in classes
}
print(json.dumps(payload, sort_keys=True))
"""


@pytest.fixture(scope="module")
def fresh_probe() -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OPENBROWSER_SETUP_LOGGING"] = "false"
    env["OPENBROWSER_DISABLE_TELEMETRY"] = "true"
    env["ANONYMIZED_TELEMETRY"] = "false"
    completed = subprocess.run(
        [sys.executable, "-B", "-c", _PROBE],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def _complete(payload: dict[str, bool]) -> set[str]:
    return {name for name, complete in payload.items() if complete}


def test_every_event_class_defers_model_build(fresh_probe: dict[str, Any]) -> None:
    assert fresh_probe["classes"] == list(EVENT_CLASSES)
    assert fresh_probe["defer_build"] == {name: True for name in EVENT_CLASSES}
    assert fresh_probe["browser_launch_result"] == {"is_event": False, "defer_build": False}


def test_fresh_import_defers_builds_until_the_explicit_owner_modules(
    fresh_probe: dict[str, Any],
) -> None:
    assert fresh_probe["session_module_loaded"] is False
    assert _complete(fresh_probe["complete_after_events"]) == {"ElementSelectedEvent"}
    assert _complete(fresh_probe["complete_after_tools"]) == {
        "ElementSelectedEvent",
        *REBUILT_BY_TOOLS,
    }
    assert _complete(fresh_probe["complete_after_default_action"]) == {
        "ElementSelectedEvent",
        *REBUILT_BY_TOOLS,
        *REBUILT_BY_DEFAULT_ACTION,
    }


def test_direct_generic_result_types_are_preserved(fresh_probe: dict[str, Any]) -> None:
    expected = {name: RESULT_TYPES.get(name, "None") for name in EVENT_CLASSES}
    assert fresh_probe["result_types"] == expected


def test_validation_errors_and_json_serialization_are_preserved() -> None:
    from openbrowser.browser.events import (
        BrowserStartEvent,
        ClickElementEvent,
        DialogOpenedEvent,
        NavigateToUrlEvent,
        SendKeysEvent,
        TabCreatedEvent,
        WaitEvent,
    )

    event = NavigateToUrlEvent(
        url="https://example.com",
        wait_until="domcontentloaded",
        new_tab=True,
    )
    payload = event.model_dump(mode="json")
    assert payload["url"] == "https://example.com"
    assert payload["wait_until"] == "domcontentloaded"
    assert payload["new_tab"] is True
    assert payload["event_type"] == "NavigateToUrlEvent"
    assert NavigateToUrlEvent.model_validate_json(event.model_dump_json()).url == event.url

    start = BrowserStartEvent(launch_options={"mode": "test"})
    assert start.model_dump(mode="json")["launch_options"] == {"mode": "test"}

    invalid_factories = (
        ("NavigateToUrlEvent", lambda: NavigateToUrlEvent(url=1), "url"),
        ("BrowserStartEvent", lambda: BrowserStartEvent(cdp_url=1), "cdp_url"),
        ("TabCreatedEvent", lambda: TabCreatedEvent(target_id=1, url=1), "target_id"),
        (
            "DialogOpenedEvent",
            lambda: DialogOpenedEvent(dialog_type=1, message=1, url=1),
            "dialog_type",
        ),
        ("SendKeysEvent", lambda: SendKeysEvent(keys=1), "keys"),
        ("ClickElementEvent", lambda: ClickElementEvent(), "node"),
        ("WaitEvent", lambda: WaitEvent(seconds="slow"), "seconds"),
    )
    for _name, factory, first_location in invalid_factories:
        with pytest.raises(ValidationError) as raised:
            factory()
        errors = raised.value.errors(include_url=False)
        assert errors
        assert errors[0]["loc"][0] == first_location


def test_schema_generation_keeps_its_existing_fail_closed_behavior() -> None:
    import openbrowser.browser.events as events

    for name in EVENT_CLASSES:
        event_class = getattr(events, name)
        with pytest.raises(PydanticInvalidForJsonSchema, match="BaseException"):
            event_class.model_json_schema()
