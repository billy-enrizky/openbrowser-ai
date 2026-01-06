"""Telemetry service for tracking agent performance."""

from __future__ import annotations

import logging
import os
import uuid
from typing import Optional

from openbrowser.telemetry.views import AgentTelemetryEvent, StepTelemetryEvent

logger = logging.getLogger(__name__)


class ProductTelemetry:
    """
    Telemetry service for tracking agent runs and performance.
    Uses PostHog for analytics when available.
    """

    def __init__(
        self,
        api_key: str | None = None,
        enabled: bool | None = None,
    ):
        """
        Initialize telemetry service.

        Args:
            api_key: PostHog API key (or POSTHOG_API_KEY env var)
            enabled: Whether telemetry is enabled (or OPENBROWSER_AI_TELEMETRY_ENABLED env var)
        """
        self._enabled = enabled if enabled is not None else self._is_enabled()
        self._api_key = api_key or os.environ.get("POSTHOG_API_KEY")
        self._posthog = None
        self._user_id = self._get_user_id()

        if self._enabled and self._api_key:
            self._init_posthog()

    def _is_enabled(self) -> bool:
        """Check if telemetry is enabled via environment variable."""
        env_value = os.environ.get("OPENBROWSER_AI_TELEMETRY_ENABLED", "true").lower()
        return env_value not in ("false", "0", "no", "off")

    def _get_user_id(self) -> str:
        """Get or generate a unique user ID."""
        # Try to get from environment or generate a new one
        user_id = os.environ.get("OPENBROWSER_AI_USER_ID")
        if not user_id:
            user_id = str(uuid.uuid4())
        return user_id

    def _init_posthog(self) -> None:
        """Initialize PostHog client."""
        try:
            import posthog

            posthog.project_api_key = self._api_key
            posthog.host = "https://app.posthog.com"
            self._posthog = posthog
            logger.debug("PostHog telemetry initialized")
        except ImportError:
            logger.debug("PostHog not installed, telemetry disabled")
            self._enabled = False

    @property
    def enabled(self) -> bool:
        """Check if telemetry is enabled."""
        return self._enabled and self._posthog is not None

    def track_agent_run(self, event: AgentTelemetryEvent) -> None:
        """
        Track an agent run event.

        Args:
            event: Agent telemetry event to track
        """
        if not self.enabled:
            return

        try:
            self._posthog.capture(
                distinct_id=self._user_id,
                event="agent_run",
                properties={
                    "run_id": event.run_id,
                    "task_length": len(event.task),
                    "total_steps": event.total_steps,
                    "total_duration_seconds": event.total_duration_seconds,
                    "is_successful": event.is_successful,
                    "is_done": event.is_done,
                    "total_errors": event.total_errors,
                    "llm_provider": event.llm_provider,
                    "llm_model": event.llm_model,
                    "total_llm_calls": event.total_llm_calls,
                    "total_input_tokens": event.total_input_tokens,
                    "total_output_tokens": event.total_output_tokens,
                    "total_cost": event.total_cost,
                    "total_actions": event.total_actions,
                    "action_summary": event.action_summary,
                    "browser_headless": event.browser_headless,
                    "use_vision": event.use_vision,
                    "version": event.version,
                },
            )
            logger.debug(f"Tracked agent run: {event.run_id}")
        except Exception as e:
            logger.warning(f"Failed to track agent run: {e}")

    def track_step(self, event: StepTelemetryEvent) -> None:
        """
        Track an individual step event.

        Args:
            event: Step telemetry event to track
        """
        if not self.enabled:
            return

        try:
            self._posthog.capture(
                distinct_id=self._user_id,
                event="agent_step",
                properties={
                    "run_id": event.run_id,
                    "step_number": event.step_number,
                    "action_name": event.action_name,
                    "duration_seconds": event.duration_seconds,
                    "is_successful": event.is_successful,
                    "has_error": event.error is not None,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to track step: {e}")

    def track_error(self, run_id: str, error: str) -> None:
        """
        Track an error event.

        Args:
            run_id: Run identifier
            error: Error message
        """
        if not self.enabled:
            return

        try:
            self._posthog.capture(
                distinct_id=self._user_id,
                event="agent_error",
                properties={
                    "run_id": run_id,
                    "error": error[:500],  # Truncate long errors
                },
            )
        except Exception as e:
            logger.warning(f"Failed to track error: {e}")

    def flush(self) -> None:
        """Flush pending events."""
        if self._posthog:
            try:
                self._posthog.flush()
            except Exception as e:
                logger.warning(f"Failed to flush telemetry: {e}")

    def shutdown(self) -> None:
        """Shutdown telemetry service."""
        if self._posthog:
            try:
                self._posthog.shutdown()
            except Exception as e:
                logger.warning(f"Failed to shutdown telemetry: {e}")

