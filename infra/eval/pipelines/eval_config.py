"""Evaluation configuration for OpenBrowser-AI benchmarks."""

import logging
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Configuration for a benchmark evaluation run."""

    # Project identifier (benchmarking, finetuning, flow_matching)
    project: str = "benchmarking"

    # Datasets to evaluate
    datasets: list[str] = field(
        default_factory=lambda: ["stress_tests"]
    )

    # LLM models to test
    models: list[str] = field(
        default_factory=lambda: ["gemini-2.5-flash"]
    )

    # Agent types to compare
    agent_types: list[Literal["Agent", "CodeAgent"]] = field(
        default_factory=lambda: ["Agent", "CodeAgent"]
    )

    # Max tasks per dataset (0 = all)
    max_tasks: int = 0

    # Max steps per agent run
    max_steps: int = 50

    # Max failures before giving up on a task
    max_failures: int = 5

    # Whether to run headless browser
    headless: bool = True

    # Output directory for results
    output_dir: str = "results"

    # S3 bucket for uploading results (empty = local only)
    results_bucket: str = ""

    # S3 bucket for downloading datasets (empty = local only)
    data_bucket: str = ""

    # AWS region
    aws_region: str = "ca-central-1"

    # Delay between tasks (seconds)
    task_delay: float = 2.0

    # Run ID (auto-generated if empty)
    run_id: str = ""

    # Port for FormFactory Flask server (5050 to avoid macOS AirPlay on 5000)
    formfactory_port: int = 5050

    # Hostname for WebArena Docker containers (localhost or remote EC2 IP)
    webarena_hostname: str = "localhost"

    def validate(self):
        """Validate configuration."""
        valid_datasets = ["stress_tests", "mind2web", "formfactory", "webarena"]
        for ds in self.datasets:
            if ds not in valid_datasets:
                raise ValueError(
                    f"Unknown dataset '{ds}'. Valid: {valid_datasets}"
                )

        valid_agents = ["Agent", "CodeAgent"]
        for at in self.agent_types:
            if at not in valid_agents:
                raise ValueError(
                    f"Unknown agent type '{at}'. Valid: {valid_agents}"
                )

        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")

        logger.info(
            f"EvalConfig validated: project={self.project}, "
            f"datasets={self.datasets}, models={self.models}, "
            f"agent_types={self.agent_types}, max_tasks={self.max_tasks}"
        )
