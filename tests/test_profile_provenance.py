"""Tests for effective profile provenance and redaction."""

import json
import math

from openbrowser.browser.profile_provenance import (
    VALUE_FIELDS,
    ProfileProvenance,
    ProfileValueSource,
)


def test_value_fields_record_the_source_that_last_set_them():
    provenance = ProfileProvenance()
    provenance.record_mapping(
        {"user_data_dir": "/tmp/one", "headless": True, "downloads_path": "/tmp/downloads"},
        ProfileValueSource.DEFAULT,
    )
    provenance.record("headless", False, ProfileValueSource.ENVIRONMENT)

    result = provenance.to_dict()

    assert result["user_data_dir"] == {"value": "/tmp/one", "source": "default"}
    assert result["downloads_path"] == {"value": "/tmp/downloads", "source": "default"}
    assert result["headless"] == {"value": False, "source": "environment"}


def test_record_mapping_covers_every_exposed_value_field():
    values = {
        "user_data_dir": "/tmp/profile",
        "headless": False,
        "channel": "chromium",
        "executable_path": "/tmp/chrome",
        "downloads_path": "/tmp/downloads",
        "keep_alive": True,
        "wait_between_actions": 0.5,
        "device_scale_factor": 1.0,
        "disable_security": False,
    }

    provenance = ProfileProvenance()
    provenance.record_mapping(values, ProfileValueSource.CONFIG)

    result = provenance.to_dict()
    assert set(result) == set(VALUE_FIELDS)
    assert all(result[name]["source"] == "config" for name in VALUE_FIELDS)


def test_sensitive_fields_expose_only_mode_and_source():
    provenance = ProfileProvenance()
    provenance.record(
        "storage_state",
        {"cookies": [{"value": "secret-cookie"}]},
        ProfileValueSource.CONFIG,
    )
    provenance.record(
        "proxy",
        {"server": "http://user:secret-password@proxy.test"},
        ProfileValueSource.CONFIG,
    )
    provenance.record("args", ["--token=secret-token"], ProfileValueSource.MANAGED)

    result = provenance.to_dict()

    assert result["storage_state"] == {"mode": "memory", "source": "config"}
    assert result["proxy"] == {"mode": "configured", "source": "config"}
    assert result["args"] == {"mode": "configured", "source": "managed", "count": 1}
    assert "secret" not in str(result)


def test_storage_state_modes():
    provenance = ProfileProvenance()
    provenance.record("storage_state", "/tmp/state.json", ProfileValueSource.CONFIG)
    assert provenance.to_dict()["storage_state"] == {"mode": "file", "source": "config"}
    provenance.record("storage_state", None, ProfileValueSource.DEFAULT)
    assert provenance.to_dict()["storage_state"] == {"mode": "none", "source": "default"}


def test_changed_managed_fields_override_the_earlier_source():
    provenance = ProfileProvenance()
    before = {"user_data_dir": "/tmp/configured"}
    after = {
        "user_data_dir": "/tmp/managed",
        "storage_state": "/tmp/managed/state.json",
    }
    provenance.record_mapping(before, ProfileValueSource.CONFIG)

    provenance.record_changes(before, after, ProfileValueSource.MANAGED)

    result = provenance.to_dict()
    assert result["user_data_dir"] == {"value": "/tmp/managed", "source": "managed"}
    assert result["storage_state"] == {"mode": "file", "source": "managed"}


def test_long_values_are_bounded():
    provenance = ProfileProvenance()
    provenance.record("user_data_dir", "x" * 400, ProfileValueSource.CONFIG)

    value = provenance.to_dict()["user_data_dir"]["value"]

    assert len(value) == 303
    assert value.endswith("...")


def test_refresh_keeps_the_source_and_updates_the_normalized_value():
    provenance = ProfileProvenance()
    provenance.record("user_data_dir", "~/profile", ProfileValueSource.CONFIG)

    provenance.refresh("user_data_dir", "/home/user/profile")

    assert provenance.to_dict()["user_data_dir"] == {
        "value": "/home/user/profile",
        "source": "config",
    }


def test_defaults_fill_only_fields_that_no_higher_layer_set():
    provenance = ProfileProvenance()
    provenance.record("headless", False, ProfileValueSource.CONFIG)

    provenance.record_defaults({"headless": True, "channel": None, "proxy": {}})

    result = provenance.to_dict()
    assert result["headless"] == {"value": False, "source": "config"}
    assert result["channel"] == {"value": None, "source": "default"}
    assert result["proxy"] == {"mode": "none", "source": "default"}


def test_non_finite_float_values_are_json_safe():
    provenance = ProfileProvenance()
    provenance.record("wait_between_actions", math.inf, ProfileValueSource.CONFIG)
    provenance.record("device_scale_factor", math.nan, ProfileValueSource.CONFIG)

    result = provenance.to_dict()

    assert result["wait_between_actions"]["value"] == "inf"
    assert result["device_scale_factor"]["value"] == "nan"
    assert json.loads(json.dumps(result))["wait_between_actions"]["value"] == "inf"


def test_malformed_args_records_a_mode_without_exposing_the_value():
    provenance = ProfileProvenance()

    provenance.record("args", "--token=secret-token", ProfileValueSource.CONFIG)

    assert provenance.to_dict()["args"] == {"mode": "configured", "source": "config"}
    assert "secret-token" not in str(provenance.to_dict())


def test_empty_proxy_is_reported_as_none():
    provenance = ProfileProvenance()

    provenance.record("proxy", {}, ProfileValueSource.CONFIG)

    assert provenance.to_dict()["proxy"] == {"mode": "none", "source": "config"}


def test_enum_values_are_stored_as_plain_strings():
    from openbrowser.browser.profile import BrowserChannel

    provenance = ProfileProvenance()

    provenance.record("channel", BrowserChannel.CHROMIUM, ProfileValueSource.CONFIG)

    value = provenance.to_dict()["channel"]["value"]
    assert value == "chromium"
    assert type(value) is str


def test_empty_validated_proxy_is_reported_as_none():
    from openbrowser.browser.profile import ProxySettings

    provenance = ProfileProvenance()
    provenance.record("proxy", {}, ProfileValueSource.CONFIG)

    provenance.refresh("proxy", ProxySettings())

    assert provenance.to_dict()["proxy"] == {"mode": "none", "source": "config"}
