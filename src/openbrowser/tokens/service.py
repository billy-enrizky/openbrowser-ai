"""Token cost tracking service."""

from __future__ import annotations

import logging
from typing import Optional

from openbrowser.tokens.views import ModelPricing, TokenUsage, CumulativeTokenUsage

logger = logging.getLogger(__name__)

# Pricing data (as of late 2024 - update as needed)
MODEL_PRICING: dict[str, ModelPricing] = {
    # OpenAI
    "gpt-4o": ModelPricing(provider="openai", model="gpt-4o", input_cost_per_1k=0.0025, output_cost_per_1k=0.01),
    "gpt-4o-mini": ModelPricing(provider="openai", model="gpt-4o-mini", input_cost_per_1k=0.00015, output_cost_per_1k=0.0006),
    "gpt-4-turbo": ModelPricing(provider="openai", model="gpt-4-turbo", input_cost_per_1k=0.01, output_cost_per_1k=0.03),
    "gpt-3.5-turbo": ModelPricing(provider="openai", model="gpt-3.5-turbo", input_cost_per_1k=0.0005, output_cost_per_1k=0.0015),
    "o3": ModelPricing(provider="openai", model="o3", input_cost_per_1k=0.01, output_cost_per_1k=0.04),

    # Anthropic
    "claude-3-opus-20240229": ModelPricing(provider="anthropic", model="claude-3-opus-20240229", input_cost_per_1k=0.015, output_cost_per_1k=0.075),
    "claude-3-5-sonnet-20241022": ModelPricing(provider="anthropic", model="claude-3-5-sonnet-20241022", input_cost_per_1k=0.003, output_cost_per_1k=0.015),
    "claude-sonnet-4-20250514": ModelPricing(provider="anthropic", model="claude-sonnet-4-20250514", input_cost_per_1k=0.003, output_cost_per_1k=0.015),

    # Google
    "gemini-2.0-flash": ModelPricing(provider="google", model="gemini-2.0-flash", input_cost_per_1k=0.000075, output_cost_per_1k=0.0003),
    "gemini-1.5-pro": ModelPricing(provider="google", model="gemini-1.5-pro", input_cost_per_1k=0.00125, output_cost_per_1k=0.005),
    "gemini-1.5-flash": ModelPricing(provider="google", model="gemini-1.5-flash", input_cost_per_1k=0.000075, output_cost_per_1k=0.0003),

    # Groq
    "llama-3.3-70b-versatile": ModelPricing(provider="groq", model="llama-3.3-70b-versatile", input_cost_per_1k=0.00059, output_cost_per_1k=0.00079),
    "llama-3.1-70b-versatile": ModelPricing(provider="groq", model="llama-3.1-70b-versatile", input_cost_per_1k=0.00059, output_cost_per_1k=0.00079),
    "mixtral-8x7b-32768": ModelPricing(provider="groq", model="mixtral-8x7b-32768", input_cost_per_1k=0.00024, output_cost_per_1k=0.00024),
}


class TokenCost:
    """
    Token cost tracking service.
    Tracks usage and calculates costs for LLM API calls.
    """

    def __init__(self, model: str | None = None):
        """
        Initialize token cost tracker.

        Args:
            model: Model name for pricing lookup
        """
        self.model = model
        self.cumulative = CumulativeTokenUsage()
        self._pricing = self._get_pricing(model) if model else None

    def _get_pricing(self, model: str) -> Optional[ModelPricing]:
        """Get pricing for a model."""
        # Try exact match first
        if model in MODEL_PRICING:
            return MODEL_PRICING[model]

        # Try partial match
        for key, pricing in MODEL_PRICING.items():
            if key in model or model in key:
                return pricing

        logger.warning(f"No pricing found for model: {model}")
        return None

    def set_model(self, model: str) -> None:
        """Set the current model for pricing."""
        self.model = model
        self._pricing = self._get_pricing(model)

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        image_count: int = 0,
    ) -> TokenUsage:
        """
        Calculate cost for token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            image_count: Number of images processed

        Returns:
            TokenUsage with calculated costs
        """
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            image_count=image_count,
        )

        if self._pricing:
            usage.input_cost = (input_tokens / 1000) * self._pricing.input_cost_per_1k
            usage.output_cost = (output_tokens / 1000) * self._pricing.output_cost_per_1k
            usage.image_cost = image_count * self._pricing.image_cost
            usage.total_cost = usage.input_cost + usage.output_cost + usage.image_cost

        # Add to cumulative
        self.cumulative.add(usage)

        return usage

    def track(
        self,
        input_tokens: int,
        output_tokens: int,
        image_count: int = 0,
    ) -> float:
        """
        Track token usage and return total cost.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            image_count: Number of images processed

        Returns:
            Total cost in USD
        """
        usage = self.calculate_cost(input_tokens, output_tokens, image_count)
        return usage.total_cost

    def get_cumulative(self) -> CumulativeTokenUsage:
        """Get cumulative usage across all calls."""
        return self.cumulative

    def reset(self) -> None:
        """Reset cumulative usage."""
        self.cumulative = CumulativeTokenUsage()

    def get_summary(self) -> str:
        """Get a summary of token usage and costs."""
        return (
            f"Token Usage Summary:\n"
            f"  Calls: {self.cumulative.call_count}\n"
            f"  Input tokens: {self.cumulative.total_input_tokens:,}\n"
            f"  Output tokens: {self.cumulative.total_output_tokens:,}\n"
            f"  Total tokens: {self.cumulative.total_tokens:,}\n"
            f"  Images: {self.cumulative.total_images}\n"
            f"  Total cost: ${self.cumulative.total_cost:.4f}"
        )

