"""Pydantic models for evaluation results."""

import logging
from datetime import datetime

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TaskResult(BaseModel):
    """Result of a single task execution."""

    # Task metadata
    task_id: str
    task_name: str
    dataset: str
    category: str = ""
    instruction: str = ""
    ground_truth: str = ""

    # Execution metadata
    agent_type: str  # "Agent" or "CodeAgent"
    model: str
    project: str = "benchmarking"
    run_id: str = ""

    # Results
    success: bool = False
    execution_time: float = 0.0
    steps_taken: int = 0
    final_output: str | None = None
    error_message: str | None = None
    output_files: list[str] = Field(default_factory=list)

    # Timestamps
    started_at: datetime | None = None
    completed_at: datetime | None = None


class RunSummary(BaseModel):
    """Summary of a complete evaluation run."""

    # Run metadata
    run_id: str
    project: str
    started_at: datetime
    completed_at: datetime | None = None

    # Configuration
    datasets: list[str]
    models: list[str]
    agent_types: list[str]
    max_tasks: int = 0
    max_steps: int = 50

    # Aggregate results
    total_tasks: int = 0
    total_successes: int = 0
    total_failures: int = 0
    total_errors: int = 0
    avg_execution_time: float = 0.0
    success_rate: float = 0.0

    # Per-agent breakdown
    agent_summaries: dict[str, dict] = Field(default_factory=dict)

    # Per-dataset breakdown
    dataset_summaries: dict[str, dict] = Field(default_factory=dict)

    # Per-model breakdown
    model_summaries: dict[str, dict] = Field(default_factory=dict)

    # All individual results
    results: list[TaskResult] = Field(default_factory=list)

    def compute_summaries(self):
        """Compute summary statistics from individual results."""
        if not self.results:
            return

        self.total_tasks = len(self.results)
        self.total_successes = sum(1 for r in self.results if r.success)
        self.total_failures = sum(1 for r in self.results if not r.success and not r.error_message)
        self.total_errors = sum(1 for r in self.results if r.error_message)
        self.success_rate = self.total_successes / self.total_tasks if self.total_tasks > 0 else 0.0
        self.avg_execution_time = sum(r.execution_time for r in self.results) / self.total_tasks

        # Per-agent
        for agent_type in set(r.agent_type for r in self.results):
            agent_results = [r for r in self.results if r.agent_type == agent_type]
            successes = sum(1 for r in agent_results if r.success)
            self.agent_summaries[agent_type] = {
                "total": len(agent_results),
                "successes": successes,
                "success_rate": successes / len(agent_results) if agent_results else 0.0,
                "avg_time": sum(r.execution_time for r in agent_results) / len(agent_results),
                "avg_steps": sum(r.steps_taken for r in agent_results) / len(agent_results),
            }

        # Per-dataset
        for dataset in set(r.dataset for r in self.results):
            ds_results = [r for r in self.results if r.dataset == dataset]
            successes = sum(1 for r in ds_results if r.success)
            self.dataset_summaries[dataset] = {
                "total": len(ds_results),
                "successes": successes,
                "success_rate": successes / len(ds_results) if ds_results else 0.0,
                "avg_time": sum(r.execution_time for r in ds_results) / len(ds_results),
            }

        # Per-model
        for model in set(r.model for r in self.results):
            model_results = [r for r in self.results if r.model == model]
            successes = sum(1 for r in model_results if r.success)
            self.model_summaries[model] = {
                "total": len(model_results),
                "successes": successes,
                "success_rate": successes / len(model_results) if model_results else 0.0,
                "avg_time": sum(r.execution_time for r in model_results) / len(model_results),
            }
