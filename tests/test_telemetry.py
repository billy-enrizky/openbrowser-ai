"""Comprehensive tests for openbrowser.telemetry.service and telemetry.views modules.

Covers: ProductTelemetry, BaseTelemetryEvent, AgentTelemetryEvent,
MCPClientTelemetryEvent, MCPServerTelemetryEvent, CLITelemetryEvent.
"""

import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from openbrowser.telemetry.views import (
    AgentTelemetryEvent,
    BaseTelemetryEvent,
    CLITelemetryEvent,
    MCPClientTelemetryEvent,
    MCPServerTelemetryEvent,
    _cached_is_docker,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BaseTelemetryEvent (via subclasses)
# ---------------------------------------------------------------------------


class TestBaseTelemetryEvent:
    def test_properties_includes_is_docker(self):
        event = CLITelemetryEvent(
            version="0.1.0",
            action="start",
            mode="interactive",
        )
        props = event.properties
        assert "is_docker" in props
        assert isinstance(props["is_docker"], bool)

    def test_properties_excludes_name(self):
        event = CLITelemetryEvent(
            version="0.1.0",
            action="start",
            mode="interactive",
        )
        props = event.properties
        assert "name" not in props


# ---------------------------------------------------------------------------
# AgentTelemetryEvent
# ---------------------------------------------------------------------------


class TestAgentTelemetryEvent:
    def _make_event(self):
        return AgentTelemetryEvent(
            task="Search for something",
            model="gpt-4",
            model_provider="openai",
            max_steps=10,
            max_actions_per_step=5,
            use_vision=True,
            version="0.1.0",
            source="cli",
            cdp_url=None,
            agent_type=None,
            action_errors=[None, "timeout"],
            action_history=[None, [{"action": "click"}]],
            urls_visited=["https://example.com"],
            steps=5,
            total_input_tokens=1000,
            total_output_tokens=500,
            prompt_cached_tokens=100,
            total_tokens=1500,
            total_duration_seconds=30.5,
            success=True,
            final_result_response="Done",
            error_message=None,
        )

    def test_default_name(self):
        event = self._make_event()
        assert event.name == "agent_event"

    def test_properties(self):
        event = self._make_event()
        props = event.properties
        assert props["task"] == "Search for something"
        assert props["model"] == "gpt-4"
        assert props["steps"] == 5
        assert props["success"] is True
        assert "is_docker" in props
        assert "name" not in props

    def test_code_agent_type(self):
        event = AgentTelemetryEvent(
            task="task",
            model="model",
            model_provider="provider",
            max_steps=1,
            max_actions_per_step=1,
            use_vision=False,
            version="0.1.0",
            source="api",
            cdp_url="ws://localhost:9222",
            agent_type="code",
            action_errors=[],
            action_history=[],
            urls_visited=[],
            steps=0,
            total_input_tokens=0,
            total_output_tokens=0,
            prompt_cached_tokens=0,
            total_tokens=0,
            total_duration_seconds=0.0,
            success=None,
            final_result_response=None,
            error_message=None,
        )
        assert event.agent_type == "code"


# ---------------------------------------------------------------------------
# MCPClientTelemetryEvent
# ---------------------------------------------------------------------------


class TestMCPClientTelemetryEvent:
    def test_default_name(self):
        event = MCPClientTelemetryEvent(
            server_name="test-server",
            command="connect",
            tools_discovered=5,
            version="0.1.0",
            action="connect",
        )
        assert event.name == "mcp_client_event"

    def test_properties(self):
        event = MCPClientTelemetryEvent(
            server_name="test-server",
            command="connect",
            tools_discovered=5,
            version="0.1.0",
            action="tool_call",
            tool_name="navigate",
            duration_seconds=1.5,
            error_message=None,
        )
        props = event.properties
        assert props["server_name"] == "test-server"
        assert props["tool_name"] == "navigate"
        assert props["duration_seconds"] == 1.5
        assert "is_docker" in props

    def test_optional_fields(self):
        event = MCPClientTelemetryEvent(
            server_name="s",
            command="c",
            tools_discovered=0,
            version="v",
            action="disconnect",
        )
        assert event.tool_name is None
        assert event.duration_seconds is None
        assert event.error_message is None


# ---------------------------------------------------------------------------
# MCPServerTelemetryEvent
# ---------------------------------------------------------------------------


class TestMCPServerTelemetryEvent:
    def test_default_name(self):
        event = MCPServerTelemetryEvent(
            version="0.1.0",
            action="start",
        )
        assert event.name == "mcp_server_event"

    def test_properties(self):
        event = MCPServerTelemetryEvent(
            version="0.1.0",
            action="tool_call",
            tool_name="screenshot",
            duration_seconds=0.5,
            error_message=None,
            parent_process_cmdline="python agent.py",
        )
        props = event.properties
        assert props["action"] == "tool_call"
        assert props["tool_name"] == "screenshot"
        assert props["parent_process_cmdline"] == "python agent.py"
        assert "is_docker" in props

    def test_optional_fields(self):
        event = MCPServerTelemetryEvent(
            version="v",
            action="stop",
        )
        assert event.tool_name is None
        assert event.duration_seconds is None
        assert event.error_message is None
        assert event.parent_process_cmdline is None


# ---------------------------------------------------------------------------
# CLITelemetryEvent
# ---------------------------------------------------------------------------


class TestCLITelemetryEvent:
    def test_default_name(self):
        event = CLITelemetryEvent(
            version="0.1.0",
            action="start",
            mode="interactive",
        )
        assert event.name == "cli_event"

    def test_properties(self):
        event = CLITelemetryEvent(
            version="0.1.0",
            action="message_sent",
            mode="oneshot",
            model="gpt-4",
            model_provider="openai",
            duration_seconds=5.0,
            error_message=None,
        )
        props = event.properties
        assert props["action"] == "message_sent"
        assert props["mode"] == "oneshot"
        assert props["model"] == "gpt-4"
        assert "is_docker" in props

    def test_optional_fields(self):
        event = CLITelemetryEvent(
            version="v",
            action="error",
            mode="mcp_server",
        )
        assert event.model is None
        assert event.model_provider is None
        assert event.duration_seconds is None
        assert event.error_message is None


# ---------------------------------------------------------------------------
# _cached_is_docker
# ---------------------------------------------------------------------------


class TestCachedIsDocker:
    def test_returns_bool(self):
        _cached_is_docker.cache_clear()
        result = _cached_is_docker()
        assert isinstance(result, bool)

    def test_caching(self):
        _cached_is_docker.cache_clear()
        r1 = _cached_is_docker()
        r2 = _cached_is_docker()
        assert r1 == r2


# ---------------------------------------------------------------------------
# ProductTelemetry (with telemetry disabled)
# ---------------------------------------------------------------------------


class TestProductTelemetry:
    """Test ProductTelemetry using MagicMock to simulate instances
    (the real class is wrapped by @singleton so object.__new__ won't work)."""

    @staticmethod
    def _get_original_class():
        """Extract the original ProductTelemetry class from the singleton wrapper closure."""
        from openbrowser.telemetry.service import ProductTelemetry

        for cell in ProductTelemetry.__closure__:
            contents = cell.cell_contents
            if isinstance(contents, type):
                return contents
        raise RuntimeError("Could not find original ProductTelemetry class in singleton closure")

    def _make_instance(self, posthog_client=None, user_id=None):
        """Create a mock ProductTelemetry-like instance with the required attributes."""
        _OrigClass = self._get_original_class()

        instance = MagicMock(spec=[])
        instance._posthog_client = posthog_client
        instance._telemetry_disabled = posthog_client is None
        instance.debug_logging = False
        instance._curr_user_id = user_id
        # Bind the real methods from the unwrapped class to the mock
        instance.capture = lambda event: _OrigClass.capture(instance, event)
        instance._direct_capture = lambda event: _OrigClass._direct_capture(instance, event)
        instance.flush = lambda: _OrigClass.flush(instance)
        return instance

    def test_capture_with_disabled_telemetry(self):
        """When telemetry is disabled (no client), capture should be a no-op."""
        instance = self._make_instance(posthog_client=None)
        event = CLITelemetryEvent(version="0.1.0", action="start", mode="interactive")
        # Should not crash
        instance.capture(event)

    def test_flush_with_no_client(self):
        instance = self._make_instance(posthog_client=None)
        # Should not crash
        instance.flush()

    def test_flush_with_mock_client(self):
        mock_client = MagicMock()
        instance = self._make_instance(posthog_client=mock_client, user_id="test-user")
        instance.flush()
        mock_client.flush.assert_called_once()

    def test_flush_handles_exception(self):
        mock_client = MagicMock()
        mock_client.flush.side_effect = RuntimeError("flush error")
        instance = self._make_instance(posthog_client=mock_client, user_id="test-user")
        # Should not raise
        instance.flush()

    def test_direct_capture_with_no_client(self):
        instance = self._make_instance(posthog_client=None)
        event = CLITelemetryEvent(version="v", action="a", mode="m")
        # Should not crash
        instance._direct_capture(event)

    def test_direct_capture_with_mock_client(self):
        mock_client = MagicMock()
        instance = self._make_instance(posthog_client=mock_client, user_id="test-user")
        # Need to bind user_id property too
        type(instance).user_id = property(lambda self: self._curr_user_id or "UNKNOWN")

        event = CLITelemetryEvent(version="v", action="a", mode="m")
        instance._direct_capture(event)
        mock_client.capture.assert_called_once()

    def test_direct_capture_handles_exception(self):
        mock_client = MagicMock()
        mock_client.capture.side_effect = RuntimeError("capture error")
        instance = self._make_instance(posthog_client=mock_client, user_id="test-user")
        type(instance).user_id = property(lambda self: self._curr_user_id or "UNKNOWN")

        event = CLITelemetryEvent(version="v", action="a", mode="m")
        # Should not raise
        instance._direct_capture(event)

    def test_user_id_property_creates_file(self, tmp_path):
        """Test the user_id property logic: creates file when missing."""
        _OrigClass = self._get_original_class()

        user_id_path = str(tmp_path / "device_id")

        instance = self._make_instance(posthog_client=None, user_id=None)
        # Bind the real user_id property from the unwrapped class
        type(instance).user_id = _OrigClass.user_id
        type(instance).USER_ID_PATH = user_id_path

        uid = instance.user_id
        assert uid is not None
        assert len(uid) > 0
        assert Path(user_id_path).exists()

    def test_user_id_property_reads_existing(self, tmp_path):
        _OrigClass = self._get_original_class()

        user_id_path = str(tmp_path / "device_id")
        Path(user_id_path).write_text("existing-user-id")

        instance = self._make_instance(posthog_client=None, user_id=None)
        type(instance).user_id = _OrigClass.user_id
        type(instance).USER_ID_PATH = user_id_path

        uid = instance.user_id
        assert uid == "existing-user-id"

    def test_user_id_property_cached(self):
        _OrigClass = self._get_original_class()

        instance = self._make_instance(posthog_client=None, user_id="cached-id")
        type(instance).user_id = _OrigClass.user_id

        uid = instance.user_id
        assert uid == "cached-id"

    def test_user_id_property_handles_permission_error(self):
        _OrigClass = self._get_original_class()

        instance = self._make_instance(posthog_client=None, user_id=None)
        type(instance).user_id = _OrigClass.user_id

        with patch("builtins.open", side_effect=PermissionError("mocked permission denied")):
            with patch("pathlib.Path.exists", return_value=True):
                uid = instance.user_id
        assert uid == "UNKNOWN_USER_ID"
