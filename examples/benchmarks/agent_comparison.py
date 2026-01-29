"""
Compare Agent vs CodeAgent performance on browser automation tasks.

This script runs the same tasks with both Agent and CodeAgent,
tracking execution time, success rate, and output files.

Usage:
    uv run examples/benchmarks/agent_comparison.py
"""

import asyncio
import csv
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv(override=True)


@dataclass
class TaskResult:
    """Result of a single task execution."""
    agent_type: str
    task_name: str
    task_instruction: str
    success: bool
    execution_time: float
    error_message: str | None = None
    output_files: list[str] = field(default_factory=list)
    steps_taken: int = 0
    final_output: str | None = None


@dataclass
class ComparisonResult:
    """Comparison results between Agent and CodeAgent."""
    task_name: str
    agent_result: TaskResult | None = None
    code_agent_result: TaskResult | None = None


# Test tasks - simpler tasks for comparison
TEST_TASKS = [
    {
        "name": "Vanilla Form - Simple",
        "instruction": """Navigate to https://openbrowser.me/openbrowser-ai/challenges/vanilla-form.html
Fill out the form with:
- First Name: John
- Last Name: Doe
- Email: john.doe@example.com
- Phone: 555-123-4567
- Address: 123 Main Street
- City: San Francisco
- State: CA
- Postal Code: 94102
- Country: United States

Submit the form and verify success message appears.""",
    },
    {
        "name": "Product Configurator - Simple",
        "instruction": """Navigate to https://openbrowser.me/openbrowser-ai/challenges/product-configurator.html
Configure a product with:
- Product Type: MacBook Pro
- Color: Space Gray
- Storage: 1TB
- RAM: 32GB
- Add Leather Case: yes
- Engraving Text: "For Billy"

Click Add to Cart and verify success.""",
    },
    {
        "name": "Data Extraction - Price Research",
        "instruction": """Navigate to https://openbrowser.me/openbrowser-ai/challenges/product-configurator.html
Extract all available:
1. Product types and their base prices
2. All color options
3. All storage options with price differences
4. All RAM options with price differences
5. All accessory options with prices

Save the extracted data to a CSV file named 'product_prices.csv' with columns:
category, option_name, price_difference

Report the total number of options extracted.""",
    },
]


async def run_agent_task(task_name: str, instruction: str) -> TaskResult:
    """Run a task using the standard Agent."""
    from openbrowser import Agent, Browser, BrowserProfile, ChatGoogle
    
    logger.info(f"[Agent] Starting task: {task_name}")
    start_time = time.time()
    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not GEMINI_API_KEY:
        return TaskResult(
            agent_type="Agent",
            task_name=task_name,
            task_instruction=instruction,
            success=False,
            execution_time=0,
            error_message="GEMINI_API_KEY not set"
        )
    
    browser = None
    try:
        llm = ChatGoogle(
            model='gemini-2.5-flash',
            temperature=0,
            api_key=GEMINI_API_KEY,
        )
        
        browser_profile = BrowserProfile(
            headless=True,  # Run headless for comparison
        )
        browser = Browser(browser_profile=browser_profile)
        
        agent = Agent(
            task=instruction,
            llm=llm,
            browser=browser,
            max_failures=3,
            max_actions_per_step=10,
        )
        
        result = await agent.run()
        execution_time = time.time() - start_time
        
        # Check for output files
        output_files = []
        if hasattr(agent, 'file_system') and agent.file_system:
            output_files = list(agent.file_system.files.keys())
        
        # Get final output
        final_output = None
        if result and hasattr(result, 'final_result'):
            final_output = str(result.final_result())
        
        # Determine success
        success = result is not None and (
            hasattr(result, 'is_done') and result.is_done() or
            len(result.history) > 0
        )
        
        steps_taken = len(result.history) if result else 0
        
        logger.info(f"[Agent] Completed task: {task_name} in {execution_time:.2f}s, success={success}")
        
        return TaskResult(
            agent_type="Agent",
            task_name=task_name,
            task_instruction=instruction,
            success=success,
            execution_time=execution_time,
            output_files=output_files,
            steps_taken=steps_taken,
            final_output=final_output
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"[Agent] Task failed: {task_name} - {e}")
        return TaskResult(
            agent_type="Agent",
            task_name=task_name,
            task_instruction=instruction,
            success=False,
            execution_time=execution_time,
            error_message=str(e)
        )
    finally:
        if browser:
            try:
                await browser.close()
            except Exception:
                pass


async def run_code_agent_task(task_name: str, instruction: str) -> TaskResult:
    """Run a task using CodeAgent."""
    from openbrowser import BrowserProfile, ChatGoogle
    from openbrowser.browser import BrowserSession
    from openbrowser.code_use import CodeAgent
    
    logger.info(f"[CodeAgent] Starting task: {task_name}")
    start_time = time.time()
    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not GEMINI_API_KEY:
        return TaskResult(
            agent_type="CodeAgent",
            task_name=task_name,
            task_instruction=instruction,
            success=False,
            execution_time=0,
            error_message="GEMINI_API_KEY not set"
        )
    
    browser_session = None
    try:
        llm = ChatGoogle(
            model='gemini-2.5-flash',
            temperature=0,
            api_key=GEMINI_API_KEY,
        )
        
        browser_profile = BrowserProfile(
            headless=True,  # Run headless for comparison
        )
        browser_session = BrowserSession(browser_profile=browser_profile)
        await browser_session.start()
        
        agent = CodeAgent(
            task=instruction,
            llm=llm,
            browser=browser_session,
            max_steps=50,
            max_failures=3,
        )
        
        result = await agent.run()
        execution_time = time.time() - start_time
        
        # Check for output files in the working directory
        output_files = []
        cwd = Path.cwd()
        for ext in ['*.csv', '*.json', '*.md']:
            for f in cwd.glob(ext):
                if f.stat().st_mtime > start_time:
                    output_files.append(f.name)
        
        # Get final output
        final_output = None
        if result and hasattr(result, 'output'):
            final_output = result.output
        
        # Determine success
        success = result is not None and (
            hasattr(result, 'success') and result.success or
            hasattr(result, 'status') and result.status == 'completed'
        )
        
        steps_taken = len(agent.complete_history) if hasattr(agent, 'complete_history') else 0
        
        logger.info(f"[CodeAgent] Completed task: {task_name} in {execution_time:.2f}s, success={success}")
        
        return TaskResult(
            agent_type="CodeAgent",
            task_name=task_name,
            task_instruction=instruction,
            success=success,
            execution_time=execution_time,
            output_files=output_files,
            steps_taken=steps_taken,
            final_output=final_output
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"[CodeAgent] Task failed: {task_name} - {e}")
        return TaskResult(
            agent_type="CodeAgent",
            task_name=task_name,
            task_instruction=instruction,
            success=False,
            execution_time=execution_time,
            error_message=str(e)
        )
    finally:
        if browser_session:
            try:
                await browser_session.close()
            except Exception:
                pass


async def run_comparison(tasks: list[dict] | None = None) -> list[ComparisonResult]:
    """Run comparison between Agent and CodeAgent on given tasks."""
    if tasks is None:
        tasks = TEST_TASKS
    
    results = []
    
    for task in tasks:
        task_name = task["name"]
        instruction = task["instruction"]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Running comparison for: {task_name}")
        logger.info(f"{'='*60}")
        
        comparison = ComparisonResult(task_name=task_name)
        
        # Run Agent
        logger.info(f"\n--- Running Agent ---")
        comparison.agent_result = await run_agent_task(task_name, instruction)
        
        # Small delay between runs
        await asyncio.sleep(2)
        
        # Run CodeAgent
        logger.info(f"\n--- Running CodeAgent ---")
        comparison.code_agent_result = await run_code_agent_task(task_name, instruction)
        
        results.append(comparison)
        
        # Print comparison summary
        print_comparison_summary(comparison)
    
    return results


def print_comparison_summary(comparison: ComparisonResult):
    """Print a summary of the comparison results."""
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPARISON SUMMARY: {comparison.task_name}")
    logger.info(f"{'='*60}")
    
    if comparison.agent_result:
        ar = comparison.agent_result
        logger.info(f"\nAgent:")
        logger.info(f"  Success: {ar.success}")
        logger.info(f"  Time: {ar.execution_time:.2f}s")
        logger.info(f"  Steps: {ar.steps_taken}")
        logger.info(f"  Output Files: {ar.output_files}")
        if ar.error_message:
            logger.info(f"  Error: {ar.error_message}")
    
    if comparison.code_agent_result:
        cr = comparison.code_agent_result
        logger.info(f"\nCodeAgent:")
        logger.info(f"  Success: {cr.success}")
        logger.info(f"  Time: {cr.execution_time:.2f}s")
        logger.info(f"  Steps: {cr.steps_taken}")
        logger.info(f"  Output Files: {cr.output_files}")
        if cr.error_message:
            logger.info(f"  Error: {cr.error_message}")
    
    # Winner determination
    if comparison.agent_result and comparison.code_agent_result:
        ar = comparison.agent_result
        cr = comparison.code_agent_result
        
        if ar.success and not cr.success:
            logger.info(f"\nWinner: Agent (CodeAgent failed)")
        elif cr.success and not ar.success:
            logger.info(f"\nWinner: CodeAgent (Agent failed)")
        elif ar.success and cr.success:
            if ar.execution_time < cr.execution_time:
                diff = cr.execution_time - ar.execution_time
                logger.info(f"\nWinner: Agent (faster by {diff:.2f}s)")
            else:
                diff = ar.execution_time - cr.execution_time
                logger.info(f"\nWinner: CodeAgent (faster by {diff:.2f}s)")
        else:
            logger.info(f"\nNo winner: Both failed")


def save_results_to_csv(results: list[ComparisonResult], filename: str = "comparison_results.csv"):
    """Save comparison results to CSV file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"comparison_results_{timestamp}.csv"
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'task_name', 
            'agent_type', 
            'success', 
            'execution_time_s', 
            'steps_taken',
            'output_files',
            'error_message'
        ])
        
        for comparison in results:
            if comparison.agent_result:
                ar = comparison.agent_result
                writer.writerow([
                    ar.task_name,
                    'Agent',
                    ar.success,
                    f"{ar.execution_time:.2f}",
                    ar.steps_taken,
                    ';'.join(ar.output_files),
                    ar.error_message or ''
                ])
            
            if comparison.code_agent_result:
                cr = comparison.code_agent_result
                writer.writerow([
                    cr.task_name,
                    'CodeAgent',
                    cr.success,
                    f"{cr.execution_time:.2f}",
                    cr.steps_taken,
                    ';'.join(cr.output_files),
                    cr.error_message or ''
                ])
    
    logger.info(f"Results saved to {filepath}")
    return filepath


async def run_single_task(agent_type: str, task_name: str, instruction: str) -> TaskResult:
    """Run a single task with specified agent type."""
    if agent_type.lower() == "agent":
        return await run_agent_task(task_name, instruction)
    elif agent_type.lower() == "codeagent":
        return await run_code_agent_task(task_name, instruction)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


async def main():
    """Main entry point for comparison."""
    logger.info("Starting Agent vs CodeAgent comparison")
    logger.info(f"Number of tasks: {len(TEST_TASKS)}")
    
    results = await run_comparison()
    
    # Save results
    csv_file = save_results_to_csv(results)
    
    # Print final summary
    logger.info(f"\n{'='*60}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*60}")
    
    agent_wins = 0
    code_agent_wins = 0
    ties = 0
    
    for r in results:
        if r.agent_result and r.code_agent_result:
            ar = r.agent_result
            cr = r.code_agent_result
            
            if ar.success and not cr.success:
                agent_wins += 1
            elif cr.success and not ar.success:
                code_agent_wins += 1
            elif ar.success and cr.success:
                if ar.execution_time < cr.execution_time:
                    agent_wins += 1
                elif cr.execution_time < ar.execution_time:
                    code_agent_wins += 1
                else:
                    ties += 1
            else:
                ties += 1
    
    logger.info(f"\nAgent wins: {agent_wins}")
    logger.info(f"CodeAgent wins: {code_agent_wins}")
    logger.info(f"Ties/Both failed: {ties}")
    logger.info(f"\nResults saved to: {csv_file}")


if __name__ == "__main__":
    asyncio.run(main())
