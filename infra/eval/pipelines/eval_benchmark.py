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
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
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
    """Create LLM instance based on model name.

    Supports:
        - gemini-*: Google Gemini models
        - gpt-*, o4-*: OpenAI models
        - claude-*: Anthropic models
        - ollama:MODEL_NAME: Local Ollama models (e.g. ollama:qwen3-8b-formfactory)
        - vllm://HOST:PORT/MODEL: vLLM OpenAI-compatible endpoint
    """
    model_lower = model.lower()

    # Ollama local models: ollama:model_name or ollama:model_name@host:port
    if model_lower.startswith("ollama:"):
        from openbrowser import ChatOllama

        parts = model[len("ollama:"):].split("@")
        ollama_model = parts[0]
        host = parts[1] if len(parts) > 1 else None
        logger.info(f"Using Ollama model: {ollama_model}, host: {host or 'default'}")
        return ChatOllama(
            model=ollama_model,
            host=host,
            ollama_options={"temperature": 0},
        )

    # vLLM OpenAI-compatible endpoint: vllm://host:port/model_name
    if model_lower.startswith("vllm://"):
        from openbrowser import ChatOpenAI

        # Parse vllm://host:port/model_path
        url_part = model[len("vllm://"):]
        if "/" in url_part:
            host_port, vllm_model = url_part.split("/", 1)
        else:
            host_port = url_part
            vllm_model = "default"
        base_url = f"http://{host_port}/v1"
        logger.info(f"Using vLLM endpoint: {base_url}, model: {vllm_model}")
        return ChatOpenAI(
            model=vllm_model,
            temperature=0,
            api_key="not-needed",
            base_url=base_url,
        )

    if "gemini" in model_lower or "google" in model_lower:
        from openbrowser import ChatGoogle

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not set")
        return ChatGoogle(model=model, temperature=0, api_key=api_key)

    elif "gpt" in model_lower or "o4" in model_lower:
        from openbrowser import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        return ChatOpenAI(model=model, temperature=0, api_key=api_key)

    elif "claude" in model_lower:
        from openbrowser import ChatAnthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        return ChatAnthropic(model=model, temperature=0, api_key=api_key)

    else:
        raise ValueError(
            f"Unknown model: {model}. "
            "Supported prefixes: gemini, gpt, o4, claude, ollama:, vllm://"
        )


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


class FormFactoryServer:
    """Manages the FormFactory Flask server lifecycle for evaluation."""

    def __init__(self, formfactory_dir: Path, port: int = 5050):
        self.formfactory_dir = formfactory_dir
        self.port = port
        self.process: subprocess.Popen | None = None

    def start(self) -> bool:
        """Start the FormFactory Flask server.

        Returns True if server started successfully.
        """
        app_py = self.formfactory_dir / "app.py"
        if not app_py.exists():
            logger.error(
                f"FormFactory app.py not found at {app_py}. "
                "Run: uv run infra/eval/scripts/download_datasets.py --datasets formfactory"
            )
            return False

        # Check if something is already running on the port
        if self._is_running():
            logger.info(f"FormFactory server already running on port {self.port}")
            return True

        logger.info(f"Starting FormFactory server on port {self.port}...")
        self.process = subprocess.Popen(
            [
                sys.executable, "-c",
                f"from app import app; app.run(host='127.0.0.1', port={self.port}, debug=False)",
            ],
            cwd=str(self.formfactory_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to be ready
        for attempt in range(30):
            time.sleep(1)
            if self._is_running():
                logger.info(f"FormFactory server ready on port {self.port}")
                return True
            if self.process.poll() is not None:
                stderr = self.process.stderr.read().decode() if self.process.stderr else ""
                logger.error(f"FormFactory server exited unexpectedly: {stderr[:500]}")
                return False

        logger.error("FormFactory server did not start within 30 seconds")
        self.stop()
        return False

    def stop(self):
        """Stop the FormFactory Flask server."""
        if self.process and self.process.poll() is None:
            logger.info("Stopping FormFactory server...")
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            logger.info("FormFactory server stopped")
            self.process = None

    def _is_running(self) -> bool:
        """Check if the server is responding."""
        try:
            urlopen(f"http://127.0.0.1:{self.port}/", timeout=2)
            return True
        except (URLError, OSError):
            return False


class WebArenaServer:
    """Manages WebArena Docker containers for evaluation.

    Handles 3 containers: shopping (7770), shopping_admin (7780), reddit (9999).
    """

    CONTAINERS = {
        "shopping": {
            "image": "webarenaimages/shopping_final_0712:latest",
            "image_alt": "shopping_final_0712",
            "port": 7770,
            "internal_port": 80,
            "needs_url_config": True,
        },
        "shopping_admin": {
            "image": "webarenaimages/shopping_admin_final_0719:latest",
            "image_alt": "shopping_admin_final_0719",
            "port": 7780,
            "internal_port": 80,
            "needs_url_config": True,
        },
        "reddit": {
            "image": "webarenaimages/postmill-populated-exposed-withimg:latest",
            "image_alt": "postmill-populated-exposed-withimg",
            "port": 9999,
            "internal_port": 80,
            "needs_url_config": False,
        },
    }

    def __init__(self, hostname: str = "localhost"):
        self.hostname = hostname

    def start(self) -> bool:
        """Start WebArena Docker containers for available images.

        Only starts containers whose images are pulled locally.
        Returns True if at least one container started successfully.
        """
        # Check Docker availability
        if not self._docker_available():
            logger.error(
                "Docker is not available. Install Docker to run WebArena evaluation."
            )
            return False

        # Determine which images are available
        available = {}
        for name, info in self.CONTAINERS.items():
            if self._image_exists(info["image"]):
                available[name] = info
            elif self._image_exists(info["image_alt"]):
                available[name] = info
            else:
                logger.warning(f"Docker image for '{name}' not found, skipping")

        if not available:
            logger.error(
                "No WebArena Docker images found. "
                "Run: uv run infra/eval/scripts/download_datasets.py --datasets webarena"
            )
            return False

        self._active_containers = list(available.keys())
        logger.info(f"Starting WebArena containers: {self._active_containers}")

        # Start containers
        for name, info in available.items():
            if self._container_running(f"webarena-{name}"):
                logger.info(f"Container webarena-{name} already running")
                continue

            # Remove existing stopped container
            self._run_docker(["rm", "-f", f"webarena-{name}"])

            logger.info(f"Starting webarena-{name} on port {info['port']}...")
            port_map = f"{info['port']}:{info['internal_port']}"
            # Use whichever image name is available
            image = info["image"] if self._image_exists(info["image"]) else info["image_alt"]
            cmd = [
                "run", "--name", f"webarena-{name}",
                "-p", port_map, "-d", image,
            ]

            if not self._run_docker(cmd):
                logger.error(f"Failed to start webarena-{name}")
                self.stop()
                return False

        # Wait for containers to be healthy
        logger.info("Waiting for WebArena containers to be ready...")
        for name, info in available.items():
            if not self._wait_for_port(info["port"], timeout=120):
                logger.error(f"webarena-{name} did not become ready on port {info['port']}")
                self.stop()
                return False
            logger.info(f"webarena-{name} ready on port {info['port']}")

        # Configure shopping sites with correct base URLs
        if "shopping" in available:
            self._configure_shopping("shopping", 7770)
        if "shopping_admin" in available:
            self._configure_shopping("shopping_admin", 7780)

        logger.info("All WebArena containers started and configured")
        return True

    def stop(self):
        """Stop and remove active WebArena containers."""
        names = getattr(self, "_active_containers", list(self.CONTAINERS.keys()))
        logger.info(f"Stopping WebArena containers: {names}")
        for name in names:
            self._run_docker(["stop", f"webarena-{name}"])
            self._run_docker(["rm", "-f", f"webarena-{name}"])
        logger.info("WebArena containers stopped")

    def _configure_shopping(self, container_name: str, port: int):
        """Run post-start URL configuration for Magento shopping sites."""
        base_url = f"http://{self.hostname}:{port}/"
        container = f"webarena-{container_name}"

        logger.info(f"Configuring {container_name} base URL to {base_url}...")

        self._run_docker([
            "exec", container,
            "/var/www/magento2/bin/magento", "setup:store-config:set",
            f"--base-url={base_url}",
        ])
        self._run_docker([
            "exec", container,
            "mysql", "-u", "magentouser", "-pMyPassword", "magentodb",
            "-e", f'UPDATE core_config_data SET value="{base_url}" '
                  f'WHERE path = "web/secure/base_url";',
        ])
        self._run_docker([
            "exec", container,
            "/var/www/magento2/bin/magento", "cache:flush",
        ])

    def _docker_available(self) -> bool:
        """Check if Docker daemon is running."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True, timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _image_exists(self, image_name: str) -> bool:
        """Check if a Docker image is loaded."""
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", image_name],
                capture_output=True, timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _container_running(self, container_name: str) -> bool:
        """Check if a container is already running."""
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
                capture_output=True, text=True, timeout=10,
            )
            return result.returncode == 0 and "true" in result.stdout.lower()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _run_docker(self, args: list[str]) -> bool:
        """Run a docker command."""
        try:
            result = subprocess.run(
                ["docker"] + args,
                capture_output=True, text=True, timeout=120,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _wait_for_port(self, port: int, timeout: int = 180) -> bool:
        """Wait for a service to respond on a port.

        Accepts any HTTP response (200, 302, etc.) as healthy.
        """
        import http.client
        for attempt in range(timeout):
            try:
                conn = http.client.HTTPConnection(self.hostname, port, timeout=3)
                conn.request("GET", "/")
                resp = conn.getresponse()
                conn.close()
                if resp.status > 0:
                    return True
            except (OSError, http.client.HTTPException):
                pass
            time.sleep(1)
            if attempt > 0 and attempt % 30 == 0:
                logger.info(f"  Still waiting for port {port}... ({attempt}s)")
        return False


async def run_evaluation(config: EvalConfig) -> RunSummary:
    """Run the full evaluation pipeline."""
    config.validate()

    run_id = config.run_id or datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    logger.info(f"Starting evaluation run: {run_id}")
    logger.info(f"Config: {config}")

    # Start FormFactory server if needed
    formfactory_server = None
    if "formfactory" in config.datasets:
        formfactory_dir = PROJECT_ROOT / "data" / "formfactory"
        formfactory_server = FormFactoryServer(formfactory_dir, port=config.formfactory_port)
        if not formfactory_server.start():
            logger.error("Failed to start FormFactory server, removing from datasets")
            config.datasets = [d for d in config.datasets if d != "formfactory"]

    # Start WebArena Docker containers if needed
    webarena_server = None
    if "webarena" in config.datasets:
        webarena_server = WebArenaServer(hostname=config.webarena_hostname)
        if not webarena_server.start():
            logger.error("Failed to start WebArena containers, removing from datasets")
            config.datasets = [d for d in config.datasets if d != "webarena"]

    started_at = datetime.now()
    all_results: list[TaskResult] = []

    try:
        for dataset_name in config.datasets:
            logger.info(f"Loading dataset: {dataset_name}")
            tasks = load_dataset(
                dataset_name,
                max_tasks=config.max_tasks,
                formfactory_port=config.formfactory_port,
                webarena_hostname=config.webarena_hostname,
            )

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
    finally:
        if formfactory_server:
            formfactory_server.stop()
        if webarena_server:
            webarena_server.stop()

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
    parser.add_argument("--formfactory-port", type=int, default=5050, help="Port for FormFactory Flask server")
    parser.add_argument("--webarena-hostname", default="localhost", help="Hostname for WebArena containers (localhost or remote IP)")
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
        formfactory_port=args.formfactory_port,
        webarena_hostname=args.webarena_hostname,
    )

    asyncio.run(run_evaluation(config))


if __name__ == "__main__":
    main()
