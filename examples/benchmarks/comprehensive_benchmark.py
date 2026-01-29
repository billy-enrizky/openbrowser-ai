"""
Comprehensive Agent vs CodeAgent Performance Comparison.

Tests both agent types on multiple tasks using browser-use.github.io stress tests.

Usage:
    uv run examples/benchmarks/comprehensive_benchmark.py
"""

import asyncio
import csv
import os
import time
import logging
from datetime import datetime
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv(override=True)

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')

# Test tasks using browser-use.github.io
TEST_TASKS = [
    {
        "name": "Vanilla Form",
        "instruction": """Navigate to https://browser-use.github.io/stress-tests/challenges/vanilla-form.html
Wait for the page to fully load.

Fill out the form with:
- Email: john.doe@example.com
- Country: United States
- Date: 2026-02-15
- Subscription Type: Basic

Click the Submit Form button.
The form is complete when you see the success message.""",
    },
    {
        "name": "Product Configurator",
        "instruction": """Navigate to https://browser-use.github.io/stress-tests/challenges/product-configurator.html
Wait for the page to fully load.

Configure a product with:
- Product Type: MacBook Pro
- Color: Space Gray
- Storage: 1TB
- RAM: 32GB

Click Add to Cart button.
The configuration is complete when you see the success message.""",
    },
    {
        "name": "Flight Booking Flow",
        "instruction": """Navigate to https://browser-use.github.io/stress-tests/challenges/flight-booking-flow.html
Wait for the page to fully load.

Step 1 - Search Flights:
- From: New York (NYC)
- To: London (LON)
- Departure Date: 2026-03-15
- Return Date: 2026-03-22
- Passengers: 2 Adults
- Class: Business
- Click Search Flights

Step 2 - Select Flight:
- Select the first available flight option
- Click Continue

Step 3 - Passenger Details:
- First Name: John
- Last Name: Doe
- Date of Birth: 1990-05-15
- Passport: AB123456
- Click Continue

Step 4 - Seat Selection:
- Select any available seat (green)
- Click Continue

Step 5 - Payment:
- Select Credit Card
- Card Number: 4111111111111111
- Expiry: 12/28
- CVV: 123
- Cardholder Name: John Doe
- Click Complete Booking

The booking is complete when you see "Booking Confirmed!" """,
    },
]


async def run_agent_task(task_name: str, instruction: str) -> dict:
    """Run a task using the standard Agent."""
    from openbrowser import Agent, Browser, BrowserProfile, ChatGoogle
    
    logger.info(f"[Agent] Starting: {task_name}")
    start_time = time.time()
    
    browser = None
    try:
        llm = ChatGoogle(model='gemini-2.5-flash', temperature=0, api_key=GEMINI_API_KEY)
        browser_profile = BrowserProfile(headless=True)
        browser = Browser(browser_profile=browser_profile)
        
        agent = Agent(task=instruction, llm=llm, browser=browser, max_failures=5, max_actions_per_step=10)
        result = await agent.run()
        
        execution_time = time.time() - start_time
        steps = len(result.history) if result else 0
        success = result is not None and result.is_done()
        final_output = str(result.final_result())[:200] if result else None
        
        logger.info(f"[Agent] Done: {task_name} in {execution_time:.2f}s, steps={steps}, success={success}")
        
        return {
            "agent_type": "Agent",
            "task_name": task_name,
            "success": success,
            "execution_time": execution_time,
            "steps": steps,
            "final_output": final_output,
            "error": None
        }
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"[Agent] Failed: {task_name} - {e}")
        return {
            "agent_type": "Agent",
            "task_name": task_name,
            "success": False,
            "execution_time": execution_time,
            "steps": 0,
            "final_output": None,
            "error": str(e)
        }
    finally:
        if browser:
            try:
                await browser.close()
            except Exception:
                pass


async def run_code_agent_task(task_name: str, instruction: str) -> dict:
    """Run a task using CodeAgent."""
    from openbrowser import BrowserProfile, ChatGoogle
    from openbrowser.browser import BrowserSession
    from openbrowser.code_use import CodeAgent
    
    logger.info(f"[CodeAgent] Starting: {task_name}")
    start_time = time.time()
    
    browser_session = None
    try:
        llm = ChatGoogle(model='gemini-2.5-flash', temperature=0, api_key=GEMINI_API_KEY)
        browser_profile = BrowserProfile(headless=True)
        browser_session = BrowserSession(browser_profile=browser_profile)
        await browser_session.start()
        
        agent = CodeAgent(task=instruction, llm=llm, browser=browser_session, max_steps=50, max_failures=5)
        result = await agent.run()
        
        execution_time = time.time() - start_time
        steps = len(agent.complete_history) if hasattr(agent, 'complete_history') else 0
        success = result is not None
        final_output = str(result.output)[:200] if result and hasattr(result, 'output') else None
        
        logger.info(f"[CodeAgent] Done: {task_name} in {execution_time:.2f}s, steps={steps}, success={success}")
        
        return {
            "agent_type": "CodeAgent",
            "task_name": task_name,
            "success": success,
            "execution_time": execution_time,
            "steps": steps,
            "final_output": final_output,
            "error": None
        }
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"[CodeAgent] Failed: {task_name} - {e}")
        return {
            "agent_type": "CodeAgent",
            "task_name": task_name,
            "success": False,
            "execution_time": execution_time,
            "steps": 0,
            "final_output": None,
            "error": str(e)
        }
    finally:
        if browser_session:
            try:
                await browser_session.close()
            except Exception:
                pass


async def main():
    logger.info("="*70)
    logger.info("COMPREHENSIVE AGENT vs CODEAGENT COMPARISON")
    logger.info("Using browser-use.github.io stress tests")
    logger.info("="*70)
    
    all_results = []
    
    for task in TEST_TASKS:
        logger.info(f"\n{'='*70}")
        logger.info(f"TASK: {task['name']}")
        logger.info("="*70)
        
        # Run Agent
        agent_result = await run_agent_task(task["name"], task["instruction"])
        all_results.append(agent_result)
        
        await asyncio.sleep(2)
        
        # Run CodeAgent
        code_agent_result = await run_code_agent_task(task["name"], task["instruction"])
        all_results.append(code_agent_result)
        
        # Print comparison
        logger.info(f"\n--- {task['name']} Results ---")
        logger.info(f"Agent:     time={agent_result['execution_time']:.2f}s, steps={agent_result['steps']}, success={agent_result['success']}")
        logger.info(f"CodeAgent: time={code_agent_result['execution_time']:.2f}s, steps={code_agent_result['steps']}, success={code_agent_result['success']}")
        
        if agent_result['success'] and code_agent_result['success']:
            if code_agent_result['execution_time'] < agent_result['execution_time']:
                speedup = agent_result['execution_time'] / code_agent_result['execution_time']
                logger.info(f"Winner: CodeAgent ({speedup:.1f}x faster)")
            else:
                speedup = code_agent_result['execution_time'] / agent_result['execution_time']
                logger.info(f"Winner: Agent ({speedup:.1f}x faster)")
        elif agent_result['success']:
            logger.info("Winner: Agent (CodeAgent failed)")
        elif code_agent_result['success']:
            logger.info("Winner: CodeAgent (Agent failed)")
        else:
            logger.info("No winner: Both failed")
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"comprehensive_comparison_{timestamp}.csv"
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['task_name', 'agent_type', 'success', 'execution_time', 'steps', 'error'])
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                'task_name': r['task_name'],
                'agent_type': r['agent_type'],
                'success': r['success'],
                'execution_time': f"{r['execution_time']:.2f}",
                'steps': r['steps'],
                'error': r['error'] or ''
            })
    
    logger.info(f"\n\nResults saved to: {csv_file}")
    
    # Final Summary
    logger.info("\n" + "="*70)
    logger.info("FINAL SUMMARY")
    logger.info("="*70)
    
    agent_results = [r for r in all_results if r['agent_type'] == 'Agent']
    code_agent_results = [r for r in all_results if r['agent_type'] == 'CodeAgent']
    
    agent_wins = 0
    code_agent_wins = 0
    
    for ar, cr in zip(agent_results, code_agent_results):
        if ar['success'] and cr['success']:
            if ar['execution_time'] < cr['execution_time']:
                agent_wins += 1
            else:
                code_agent_wins += 1
        elif ar['success']:
            agent_wins += 1
        elif cr['success']:
            code_agent_wins += 1
    
    total_agent_time = sum(r['execution_time'] for r in agent_results)
    total_code_agent_time = sum(r['execution_time'] for r in code_agent_results)
    agent_success_rate = sum(1 for r in agent_results if r['success']) / len(agent_results) * 100
    code_agent_success_rate = sum(1 for r in code_agent_results if r['success']) / len(code_agent_results) * 100
    
    logger.info(f"\n| Metric | Agent | CodeAgent |")
    logger.info(f"|--------|-------|-----------|")
    logger.info(f"| Wins | {agent_wins} | {code_agent_wins} |")
    logger.info(f"| Total Time | {total_agent_time:.2f}s | {total_code_agent_time:.2f}s |")
    logger.info(f"| Success Rate | {agent_success_rate:.0f}% | {code_agent_success_rate:.0f}% |")
    logger.info(f"| Avg Steps | {sum(r['steps'] for r in agent_results)/len(agent_results):.1f} | {sum(r['steps'] for r in code_agent_results)/len(code_agent_results):.1f} |")
    
    if total_code_agent_time > 0 and total_agent_time > 0:
        if total_code_agent_time < total_agent_time:
            speedup = total_agent_time / total_code_agent_time
            logger.info(f"\nOverall: CodeAgent is {speedup:.2f}x faster")
        else:
            speedup = total_code_agent_time / total_agent_time
            logger.info(f"\nOverall: Agent is {speedup:.2f}x faster")
    
    return csv_file


if __name__ == "__main__":
    asyncio.run(main())
