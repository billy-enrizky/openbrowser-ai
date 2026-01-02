"""Tests for the token cost tracking module.

This module provides comprehensive test coverage for the token cost
tracking system, which monitors LLM API usage and calculates costs.
It validates:

    - TokenUsage: Data structure for tracking input/output tokens
    - ModelPricing: Per-model pricing configuration
    - TokenCost: Cost calculation and cumulative tracking
    - Cost calculation with known and unknown models
    - Cumulative tracking across multiple API calls
    - Summary generation for reporting token usage

The token tracking module enables cost monitoring and optimization
of LLM usage during browser automation tasks.
"""

import pytest

from src.openbrowser.tokens import TokenCost, TokenUsage, ModelPricing


class TestTokenUsage:
    """Tests for the TokenUsage class.

    Validates token usage data structure including default values
    and custom initialization with input, output, and total tokens.
    """

    def test_token_usage_defaults(self):
        """Test TokenUsage defaults."""
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_cost == 0.0

    def test_token_usage_with_values(self):
        """Test TokenUsage with values."""
        usage = TokenUsage(
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
        )
        assert usage.input_tokens == 1000
        assert usage.output_tokens == 500
        assert usage.total_tokens == 1500


class TestModelPricing:
    """Tests for the ModelPricing class.

    Validates pricing model creation with provider, model name,
    and per-1k-token costs for input and output.
    """

    def test_model_pricing_creation(self):
        """Test ModelPricing creation."""
        pricing = ModelPricing(
            provider="openai",
            model="gpt-4o",
            input_cost_per_1k=0.0025,
            output_cost_per_1k=0.01,
        )
        assert pricing.provider == "openai"
        assert pricing.model == "gpt-4o"
        assert pricing.input_cost_per_1k == 0.0025


class TestTokenCost:
    """Tests for the TokenCost class.

    Validates cost calculation, tracking, cumulative statistics,
    reset functionality, and summary generation for token usage.
    """

    def test_token_cost_init(self):
        """Test TokenCost initialization."""
        tc = TokenCost(model="gpt-4o")
        assert tc.model == "gpt-4o"
        assert tc._pricing is not None

    def test_token_cost_unknown_model(self):
        """Test TokenCost with unknown model."""
        tc = TokenCost(model="unknown-model-12345")
        assert tc._pricing is None

    def test_calculate_cost(self):
        """Test cost calculation."""
        tc = TokenCost(model="gpt-4o")
        usage = tc.calculate_cost(input_tokens=1000, output_tokens=500)
        
        assert usage.input_tokens == 1000
        assert usage.output_tokens == 500
        assert usage.total_tokens == 1500
        assert usage.input_cost > 0
        assert usage.output_cost > 0
        assert usage.total_cost > 0

    def test_track_returns_cost(self):
        """Test track returns cost."""
        tc = TokenCost(model="gpt-4o")
        cost = tc.track(input_tokens=1000, output_tokens=500)
        assert cost > 0

    def test_cumulative_tracking(self):
        """Test cumulative tracking."""
        tc = TokenCost(model="gpt-4o")
        
        tc.track(input_tokens=1000, output_tokens=500)
        tc.track(input_tokens=2000, output_tokens=1000)
        
        cumulative = tc.get_cumulative()
        assert cumulative.total_input_tokens == 3000
        assert cumulative.total_output_tokens == 1500
        assert cumulative.call_count == 2

    def test_reset(self):
        """Test reset cumulative."""
        tc = TokenCost(model="gpt-4o")
        tc.track(input_tokens=1000, output_tokens=500)
        tc.reset()
        
        cumulative = tc.get_cumulative()
        assert cumulative.total_tokens == 0
        assert cumulative.call_count == 0

    def test_get_summary(self):
        """Test get summary."""
        tc = TokenCost(model="gpt-4o")
        tc.track(input_tokens=1000, output_tokens=500)
        
        summary = tc.get_summary()
        assert "Token Usage Summary" in summary
        assert "1,000" in summary  # Input tokens
        assert "500" in summary  # Output tokens

