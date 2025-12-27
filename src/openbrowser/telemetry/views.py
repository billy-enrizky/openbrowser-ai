"""Telemetry views and event models."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class AgentTelemetryEvent(BaseModel):
    """Telemetry event for agent runs."""

    # Run identification
    run_id: str = Field(description="Unique identifier for this run")
    task: str = Field(description="The task description")

    # Run metrics
    total_steps: int = Field(default=0, description="Total number of steps taken")
    total_duration_seconds: float = Field(default=0.0, description="Total duration in seconds")
    is_successful: Optional[bool] = Field(default=None, description="Whether the task was successful")
    is_done: bool = Field(default=False, description="Whether the task completed")

    # Error tracking
    total_errors: int = Field(default=0, description="Total number of errors encountered")
    last_error: Optional[str] = Field(default=None, description="Last error message")

    # LLM metrics
    llm_provider: str = Field(default="unknown", description="LLM provider used")
    llm_model: str = Field(default="unknown", description="LLM model used")
    total_llm_calls: int = Field(default=0, description="Total LLM API calls")
    total_input_tokens: int = Field(default=0, description="Total input tokens")
    total_output_tokens: int = Field(default=0, description="Total output tokens")
    total_cost: float = Field(default=0.0, description="Estimated total cost in USD")

    # Actions
    total_actions: int = Field(default=0, description="Total actions executed")
    action_summary: dict[str, int] = Field(default_factory=dict, description="Count of each action type")

    # Metadata
    browser_headless: bool = Field(default=True, description="Whether browser was in headless mode")
    use_vision: bool = Field(default=True, description="Whether vision was enabled")
    version: str = Field(default="0.1.0", description="OpenBrowser version")


class StepTelemetryEvent(BaseModel):
    """Telemetry event for individual steps."""

    run_id: str
    step_number: int
    action_name: str
    action_params: dict
    duration_seconds: float
    is_successful: bool
    error: Optional[str] = None

