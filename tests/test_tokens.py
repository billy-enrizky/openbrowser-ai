"""Tests for the token cost tracking module.

This module provides test coverage for the token cost tracking system.
"""

import pytest

from openbrowser.tokens.service import TokenCost


class TestTokenCost:
    """Tests for the TokenCost class."""

    def test_token_cost_init(self):
        """Test TokenCost initialization."""
        tc = TokenCost(include_cost=False)
        assert tc.include_cost is False
        assert tc.usage_history == []
        assert tc._initialized is False

    def test_token_cost_with_include_cost(self):
        """Test TokenCost with include_cost=True."""
        tc = TokenCost(include_cost=True)
        assert tc.include_cost is True

    @pytest.mark.asyncio
    async def test_token_cost_initialize(self):
        """Test TokenCost initialization (without cost tracking)."""
        tc = TokenCost(include_cost=False)
        await tc.initialize()
        assert tc._initialized is True
