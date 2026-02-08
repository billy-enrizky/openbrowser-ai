"""
Main evaluation orchestrator for OpenBrowser-AI benchmarks.

Extends examples/benchmarks/comprehensive_benchmark.py with:
- Multiple dataset support (stress_tests, mind2web, formfactory, webarena)
- Multiple model support
- Structured results with Pydantic models
- CSV/JSON output
- S3 upload support

Usage:
    uv run infra/eval/pipelines/eval_benchmark.py --datasets stress_tests --max-tasks 5
    uv run infra/eval/pipelines/eval_benchmark.py --datasets stress_tests mind2web --models gemini-2.5-flash gpt-4o
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from infra.eval.pipelines.data_loader import load_dataset
from infra.eval.pipelines.eval_config import EvalConfig
from infra.eval.pipelines.results_schema import RunSummary, TaskResult


async def run_agent_task(
    task: dict, model: str, agent_type: str, config: EvalConfig, run_id: str
) -> TaskResult:
    """Run a single task with the specified agent type and model."""
    task_id = task.get("task_id", "")
    task_name = task.get("name", "")
    instruction = task.get("instruction", "")

    logger.info(f"[{agent_type}:{model}] Starting: {task_name}")
    started_at = datetime.now()
    start_time = time.time()

    result = TaskResult(
        task_id=task_id,
        task_name=task_name,
        dataset=task.get("dataset", ""),
        category=task.get("category", ""),
        instruction=instruction,
        ground_truth=str(task.get("ground_truth", "")),
        agent_type=agent_type,
        model=model,
        project=config.project,
        run_id=run_id,
        started_at=started_at,
    )

    try:
        if agent_type == "Agent":
            result = await _run_standard_agent(result, instruction, model, config)
        elif agent_type == "CodeAgent":
            result = await _run_code_agent(result, instruction, model, config)
        else:
            result.error_message = f"Unknown agent type: {agent_type}"
    except Exception as e:
        result.execution_time = time.time() - start_time
        result.error_message = str(e)
        logger.error(f"[{agent_type}:{model}] Error on {task_name}: {e}")

    result.completed_at = datetime.now()
    if result.execution_time == 0:
        result.execution_time = time.time() - start_time

    status = "SUCCESS" if result.success else "FAIL"
    logger.info(
        f"[{agent_type}:{model}] {status}: {task_name} "
        f"({result.execution_time:.2f}s, {result.steps_taken} steps)"
    )
    return result


async def _run_standard_agent(
    result: TaskResult, instruction: str, model: str, config: EvalConfig
) -> TaskResult:
    """Run task with standard Agent."""
    from openbrowser import Agent, Browser, BrowserProfile

    llm = _get_llm(model)
    browser = None
    start_time = time.time()

    try:
        browser_profile = BrowserProfile(headless=config.headless)
        browser = Browser(browser_profile=browser_profile)

        agent = Agent(
            task=instruction,
            llm=llm,
            browser=browser,
            max_failures=config.max_failures,
            max_actions_per_step=10,
        )
        agent_result = await agent.run()

        result.execution_time = time.time() - start_time
        result.steps_taken = len(agent_result.history) if agent_result else 0
        result.success = agent_result is not None and agent_result.is_done()
        result.final_output = (
            str(agent_result.final_result())[:500] if agent_result else None
        )
    finally:
        if browser:
            try:
                await browser.close()
            except Exception:
                pass

    return result


async def _run_code_agent(
    result: TaskResult, instruction: str, model: str, config: EvalConfig
) -> TaskResult:
    """Run task with CodeAgent."""
    from openbrowser import BrowserProfile
    from openbrowser.browser import BrowserSession
    from openbrowser.code_use import CodeAgent

    llm = _get_llm(model)
    browser_session = None
    start_time = time.time()

    try:
        browser_profile = BrowserProfile(headless=config.headless)
        browser_session = BrowserSession(browser_profile=browser_profile)
        await browser_session.start()

        agent = CodeAgent(
            task=instruction,
            llm=llm,
            browser=browser_session,
            max_steps=config.max_steps,
            max_failures=config.max_failures,
        )
        agent_result = await agent.run()

        result.execution_time = time.time() - start_time
        result.steps_taken = (
            len(agent.complete_history) if hasattr(agent, "complete_history") else 0
        )
        result.success = agent_result is not None
        result.final_output = (
            str(agent_result.output)[:500]
            if agent_result and hasattr(agent_result, "output")
            else None
        )
    finally:
        if browser_session:
            try:
                await browser_session.close()
            except Exception:
                pass

    return result


def _get_llm(model: str):
    """Create LLM instance based on model name."""
    if "gemini" in model.lower() or "google" in model.lower():
        from openbrowser import ChatGoogle

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not set")
        return ChatGoogle(model=model, temperature=0, api_key=api_key)

    elif "gpt" in model.lower() or "o4" in model.lower():
        from openbrowser import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        return ChatOpenAI(model=model, temperature=0, api_key=api_key)

    elif "claude" in model.lower():
        from openbrowser import ChatAnthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        return ChatAnthropic(model=model, temperature=0, api_key=api_key)

    else:
        raise ValueError(f"Unknown model: {model}")


def save_results_csv(results: list[TaskResult], output_path: Path):
    """Save results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_id", "task_name", "dataset", "category",
                "agent_type", "model", "project", "run_id",
                "success", "execution_time", "steps_taken",
                "final_output", "error_message",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow({
                "task_id": r.task_id,
                "task_name": r.task_name,
                "dataset": r.dataset,
                "category": r.category,
                "agent_type": r.agent_type,
                "model": r.model,
                "project": r.project,
                "run_id": r.run_id,
                "success": r.success,
                "execution_time": f"{r.execution_time:.2f}",
                "steps_taken": r.steps_taken,
                "final_output": (r.final_output or "")[:200],
                "error_message": r.error_message or "",
            })

    logger.info(f"Results saved to {output_path}")


def save_results_json(summary: RunSummary, output_path: Path):
    """Save full run summary to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(summary.model_dump(mode="json"), f, indent=2, default=str)

    logger.info(f"Summary saved to {output_path}")


def upload_to_s3(local_path: Path, bucket: str, s3_key: str):
    """Upload a file to S3."""
    try:
        import boto3

        s3 = boto3.client("s3")
        s3.upload_file(str(local_path), bucket, s3_key)
        logger.info(f"Uploaded {local_path} to s3://{bucket}/{s3_key}")
    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}")


async def run_evaluation(config: EvalConfig) -> RunSummary:
    """Run the full evaluation pipeline."""
    config.validate()

    run_id = config.run_id or datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    logger.info(f"Starting evaluation run: {run_id}")
    logger.info(f"Config: {config}")

    started_at = datetime.now()
    all_results: list[TaskResult] = []

    for dataset_name in config.datasets:
        logger.info(f"Loading dataset: {dataset_name}")
        tasks = load_dataset(dataset_name, max_tasks=config.max_tasks)

        if not tasks:
            logger.warning(f"No tasks loaded for {dataset_name}, skipping")
            continue

        logger.info(f"Loaded {len(tasks)} tasks from {dataset_name}")

        for model in config.models:
            for agent_type in config.agent_types:
                logger.info(
                    f"Running {agent_type} with {model} on {dataset_name} "
                    f"({len(tasks)} tasks)"
                )

                for task in tasks:
                    result = await run_agent_task(
                        task, model, agent_type, config, run_id
                    )
                    all_results.append(result)

                    if config.task_delay > 0:
                        await asyncio.sleep(config.task_delay)

    # Build summary
    summary = RunSummary(
        run_id=run_id,
        project=config.project,
        started_at=started_at,
        completed_at=datetime.now(),
        datasets=config.datasets,
        models=config.models,
        agent_types=config.agent_types,
        max_tasks=config.max_tasks,
        max_steps=config.max_steps,
        results=all_results,
    )
    summary.compute_summaries()

    # Save locally
    output_dir = Path(config.output_dir) / run_id
    save_results_csv(all_results, output_dir / "results.csv")
    save_results_json(summary, output_dir / "summary.json")

    # Upload to S3 if configured
    if config.results_bucket:
        s3_prefix = f"{config.project}/runs/{datetime.now().strftime('%Y-%m-%d')}/{run_id}"
        upload_to_s3(output_dir / "results.csv", config.results_bucket, f"{s3_prefix}/results.csv")
        upload_to_s3(output_dir / "summary.json", config.results_bucket, f"{s3_prefix}/summary.json")

    # Print summary
    logger.info("=" * 70)
    logger.info(f"EVALUATION COMPLETE: {run_id}")
    logger.info("=" * 70)
    logger.info(f"Total tasks: {summary.total_tasks}")
    logger.info(f"Successes: {summary.total_successes}")
    logger.info(f"Failures: {summary.total_failures}")
    logger.info(f"Errors: {summary.total_errors}")
    logger.info(f"Success rate: {summary.success_rate:.1%}")
    logger.info(f"Avg execution time: {summary.avg_execution_time:.2f}s")

    for agent_type, stats in summary.agent_summaries.items():
        logger.info(
            f"  {agent_type}: {stats['successes']}/{stats['total']} "
            f"({stats['success_rate']:.1%}), avg {stats['avg_time']:.2f}s"
        )

    return summary


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="OpenBrowser-AI Evaluation Benchmark")
    parser.add_argument(
        "--datasets", nargs="+", default=["stress_tests"],
        help="Datasets to evaluate (stress_tests, mind2web, formfactory, webarena)",
    )
    parser.add_argument(
        "--models", nargs="+", default=["gemini-2.5-flash"],
        help="LLM models to test",
    )
    parser.add_argument(
        "--agent-types", nargs="+", default=["Agent", "CodeAgent"],
        help="Agent types to compare",
    )
    parser.add_argument("--max-tasks", type=int, default=0, help="Max tasks per dataset (0=all)")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps per agent run")
    parser.add_argument("--project", default="benchmarking", help="Project identifier")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--results-bucket", default="", help="S3 bucket for results")
    parser.add_argument("--no-headless", action="store_true", help="Run with visible browser")
    parser.add_argument("--run-id", default="", help="Custom run ID")
    return parser.parse_args()


def main():
    """Entry point."""
    args = parse_args()

    config = EvalConfig(
        project=args.project,
        datasets=args.datasets,
        models=args.models,
        agent_types=args.agent_types,
        max_tasks=args.max_tasks,
        max_steps=args.max_steps,
        headless=not args.no_headless,
        output_dir=args.output_dir,
        results_bucket=args.results_bucket,
        run_id=args.run_id,
    )

    asyncio.run(run_evaluation(config))


if __name__ == "__main__":
    main()
