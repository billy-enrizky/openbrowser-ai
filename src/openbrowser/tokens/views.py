"""Token cost tracking views and models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelPricing(BaseModel):
    """Pricing information for a model."""

    provider: str
    model: str
    input_cost_per_1k: float = Field(description="Cost per 1000 input tokens in USD")
    output_cost_per_1k: float = Field(description="Cost per 1000 output tokens in USD")
    image_cost: float = Field(default=0.0, description="Cost per image in USD")


class TokenUsage(BaseModel):
    """Token usage for a single LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    image_count: int = 0

    # Calculated costs
    input_cost: float = 0.0
    output_cost: float = 0.0
    image_cost: float = 0.0
    total_cost: float = 0.0


class CumulativeTokenUsage(BaseModel):
    """Cumulative token usage across all LLM calls."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_images: int = 0
    total_cost: float = 0.0
    call_count: int = 0

    def add(self, usage: TokenUsage) -> None:
        """Add usage from a single call."""
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        self.total_tokens += usage.total_tokens
        self.total_images += usage.image_count
        self.total_cost += usage.total_cost
        self.call_count += 1

