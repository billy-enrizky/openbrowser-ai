"""Shared test helpers for a frozen, ready startup config."""

from contextlib import contextmanager
from unittest.mock import patch

from openbrowser.config import Config, create_default_config
from openbrowser.config_lifecycle import (
    ConfigFilePhase,
    ConfigFileSnapshot,
    ConfigLoadResult,
)


def ready_config_result() -> ConfigLoadResult:
    """Return a stable ready result without touching the user's config path."""
    return ConfigLoadResult(
        config=create_default_config(),
        snapshot=ConfigFileSnapshot(
            path="<test-config>",
            resolved_path="<test-config>",
            phase=ConfigFilePhase.READY,
            error_code=None,
            hint=None,
            browser_start_allowed=True,
            file_version="test-config-v1",
        ),
    )


@contextmanager
def startup_config(config_dict):
    """Freeze one ready load result and its legacy config dictionary."""
    result = ready_config_result()
    with (
        patch.object(Config, "load_result", return_value=result),
        patch.object(Config, "config_dict_from_result", return_value=dict(config_dict)),
    ):
        yield result
