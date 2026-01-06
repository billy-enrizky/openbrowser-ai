"""CLI module for openbrowser-ai."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

try:
    import click
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
except ImportError:
    raise ImportError("Please install CLI dependencies: pip install click rich")

from openbrowser.agent import BrowserAgent
from openbrowser.agent.views import AgentSettings
from openbrowser.browser.session import BrowserSession
from openbrowser.browser.profile import BrowserProfile
from openbrowser.llm import get_llm_by_name

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="openbrowser")
def cli():
    """OpenBrowser - AI-powered browser automation."""
    pass


@cli.command()
@click.argument("task")
@click.option("--provider", "-p", default="openai", help="LLM provider (openai, anthropic, google, groq, ollama, openrouter, aws, azure)")
@click.option("--model", "-m", default=None, help="Model name (uses provider default if not specified)")
@click.option("--headless/--no-headless", default=True, help="Run browser in headless mode")
@click.option("--max-steps", default=10, help="Maximum number of steps")
@click.option("--vision/--no-vision", default=True, help="Use vision capabilities")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--save-gif", type=str, default=None, help="Save execution as GIF to specified path")
def run(
    task: str,
    provider: str,
    model: Optional[str],
    headless: bool,
    max_steps: int,
    vision: bool,
    verbose: bool,
    save_gif: Optional[str],
):
    """Run a browser automation task.
    
    Executes a browser automation task using the specified LLM provider and
    configuration options. The agent will navigate web pages and interact
    with elements to accomplish the given task.
    
    Args:
        task: Natural language description of the task to accomplish.
        provider: LLM provider to use (openai, anthropic, google, groq, ollama, openrouter, aws, azure).
        model: Model name to use. If not specified, uses the provider's default model.
        headless: Whether to run the browser in headless mode (no visible window).
        max_steps: Maximum number of steps the agent can take before stopping.
        vision: Whether to enable vision capabilities for screenshot analysis.
        verbose: Enable verbose logging output for debugging.
        save_gif: Optional path to save execution recording as GIF.
        
    Example:
        >>> openbrowser-ai run "Search for Python tutorials" --provider openai --headless
    """
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    console.print(Panel.fit(
        f"[bold blue]openbrowser-ai[/bold blue]\n"
        f"Task: {task}\n"
        f"Provider: {provider}\n"
        f"Headless: {headless}",
        title="Starting Agent",
    ))

    async def execute():
        browser_profile = BrowserProfile(headless=headless)
        browser_session = BrowserSession(browser_profile=browser_profile)

        try:
            await browser_session.start()

            settings = AgentSettings(
                use_vision=vision,
                max_actions_per_step=4,
            )

            agent = BrowserAgent(
                browser_session=browser_session,
                llm_provider=provider,
                llm_model=model or _get_default_model(provider),
                max_steps=max_steps,
                settings=settings,
                task=task,
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task(description="Running agent...", total=None)
                result = await agent.run(task)

            # Display results
            console.print()
            if result.get("is_done"):
                console.print(Panel.fit(
                    f"[green]Task completed successfully![/green]\n\n"
                    f"Steps: {result.get('total_steps', 0)}\n"
                    f"Result: {result.get('final_result', 'N/A')[:200]}",
                    title="Results",
                ))
            else:
                console.print(Panel.fit(
                    f"[yellow]Task did not complete[/yellow]\n\n"
                    f"Steps: {result.get('total_steps', 0)}\n"
                    f"Errors: {result.get('errors', [])}",
                    title="Results",
                ))

            # Save GIF if requested
            if save_gif:
                try:
                    from openbrowser.agent.gif import create_gif_from_screenshots
                    screenshots = result.get("screenshots", [])
                    if screenshots:
                        gif_path = create_gif_from_screenshots(screenshots, save_gif)
                        console.print(f"[green]GIF saved to: {gif_path}[/green]")
                    else:
                        console.print("[yellow]No screenshots available for GIF[/yellow]")
                except Exception as e:
                    console.print(f"[red]Failed to save GIF: {e}[/red]")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise
        finally:
            await browser_session.stop()

    asyncio.run(execute())


@cli.command()
def init():
    """Initialize openbrowser-ai configuration.
    
    Checks for available LLM provider API keys in environment variables
    and displays their configuration status. This helps users verify
    which providers are properly configured before running tasks.
    
    The following environment variables are checked:
        - OPENAI_API_KEY: OpenAI provider
        - ANTHROPIC_API_KEY: Anthropic provider
        - GOOGLE_API_KEY: Google provider
        - GROQ_API_KEY: Groq provider
        - OPENROUTER_API_KEY: OpenRouter provider
        - AZURE_OPENAI_API_KEY: Azure OpenAI provider
    """
    console.print("[blue]Initializing openbrowser-ai configuration...[/blue]")

    # Check for environment variables
    import os

    providers_status = []
    providers = [
        ("OPENAI_API_KEY", "OpenAI"),
        ("ANTHROPIC_API_KEY", "Anthropic"),
        ("GOOGLE_API_KEY", "Google"),
        ("GROQ_API_KEY", "Groq"),
        ("OPENROUTER_API_KEY", "OpenRouter"),
        ("AZURE_OPENAI_API_KEY", "Azure OpenAI"),
    ]

    for env_var, provider_name in providers:
        if os.environ.get(env_var):
            providers_status.append(f"[green]{provider_name}: Configured[/green]")
        else:
            providers_status.append(f"[yellow]{provider_name}: Not configured ({env_var})[/yellow]")

    console.print(Panel.fit(
        "\n".join(providers_status),
        title="LLM Providers",
    ))


@cli.command()
def models():
    """List available models for each provider.
    
    Displays a formatted list of commonly used models organized by
    LLM provider. This serves as a quick reference for selecting
    the appropriate model for different use cases.
    
    Providers and example models:
        - OpenAI: gpt-4o, gpt-4o-mini, gpt-4-turbo
        - Anthropic: claude-sonnet-4-20250514, claude-3-5-sonnet
        - Google: gemini-2.0-flash, gemini-1.5-pro
        - Groq: llama-3.3-70b-versatile, mixtral-8x7b
        - Ollama: llama3.2, mistral, codellama
        - OpenRouter: Various models from multiple providers
        - AWS Bedrock: Anthropic Claude models
        - Azure OpenAI: Microsoft-hosted OpenAI models
    """
    models_info = {
        "OpenAI": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "Anthropic": ["claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"],
        "Google": ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
        "Groq": ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
        "Ollama": ["llama3.2", "mistral", "codellama"],
        "OpenRouter": ["anthropic/claude-3.5-sonnet", "openai/gpt-4o", "google/gemini-pro"],
        "AWS Bedrock": ["anthropic.claude-3-5-sonnet-20241022-v2:0"],
        "Azure OpenAI": ["gpt-4o", "gpt-4-turbo"],
    }

    for provider, model_list in models_info.items():
        console.print(f"[bold]{provider}[/bold]")
        for model in model_list:
            console.print(f"  - {model}")
        console.print()


def _get_default_model(provider: str) -> str:
    """Get default model for a provider.
    
    Returns the recommended default model for a given LLM provider.
    These defaults are chosen for a balance of capability and cost.
    
    Args:
        provider: The LLM provider name (case-insensitive).
        
    Returns:
        The default model name for the provider. Falls back to 'gpt-4o'
        if the provider is not recognized.
        
    Example:
        >>> _get_default_model('anthropic')
        'claude-sonnet-4-20250514'
        >>> _get_default_model('unknown')
        'gpt-4o'
    """
    defaults = {
        "openai": "gpt-4o",
        "anthropic": "claude-sonnet-4-20250514",
        "google": "gemini-2.0-flash",
        "groq": "llama-3.3-70b-versatile",
        "ollama": "llama3.2",
        "openrouter": "anthropic/claude-3.5-sonnet",
        "aws": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "azure": "gpt-4o",
    }
    return defaults.get(provider.lower(), "gpt-4o")


def main():
    """Main entry point for CLI.
    
    This function serves as the primary entry point for the openbrowser-ai
    command-line interface. It initializes the Click command group and
    handles command routing.
    
    The CLI provides the following commands:
        - run: Execute a browser automation task
        - init: Initialize configuration and check API keys
        - models: List available models by provider
    
    Example:
        >>> # From command line:
        >>> openbrowser-ai run "Search Google for Python tutorials"
        >>> openbrowser-ai init
        >>> openbrowser-ai models
    """
    cli()


if __name__ == "__main__":
    main()

