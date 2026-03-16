"""Tests for openbrowser.tokens.service module."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openbrowser.llm.views import ChatInvokeUsage
from openbrowser.tokens.service import TokenCost, xdg_cache_home
from openbrowser.tokens.views import ModelPricing, TokenCostCalculated, TokenUsageEntry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_usage(
    prompt_tokens=100,
    completion_tokens=50,
    prompt_cached_tokens=0,
    prompt_cache_creation_tokens=None,
    prompt_image_tokens=None,
):
    return ChatInvokeUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_cached_tokens=prompt_cached_tokens,
        prompt_cache_creation_tokens=prompt_cache_creation_tokens,
        prompt_image_tokens=prompt_image_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


# ---------------------------------------------------------------------------
# xdg_cache_home tests
# ---------------------------------------------------------------------------


class TestXdgCacheHome:
    def test_returns_path(self):
        # Clear the cache to ensure fresh call
        xdg_cache_home.cache_clear()
        result = xdg_cache_home()
        assert isinstance(result, Path)

    def test_default_is_home_cache(self):
        xdg_cache_home.cache_clear()
        with patch("openbrowser.tokens.service.CONFIG") as mock_config:
            mock_config.XDG_CACHE_HOME = None
            xdg_cache_home.cache_clear()
            result = xdg_cache_home()
            assert result == Path.home() / ".cache"


# ---------------------------------------------------------------------------
# TokenCost initialization tests
# ---------------------------------------------------------------------------


class TestTokenCostInit:
    def test_default_init(self):
        with patch.dict("os.environ", {}, clear=True):
            tc = TokenCost(include_cost=False)
            assert tc.include_cost is False
            assert tc.usage_history == []
            assert tc.registered_llms == {}
            assert tc._pricing_data is None
            assert tc._initialized is False

    def test_env_var_override(self):
        with patch.dict("os.environ", {"OPENBROWSER_CALCULATE_COST": "true"}):
            tc = TokenCost(include_cost=False)
            assert tc.include_cost is True


# ---------------------------------------------------------------------------
# TokenCost.initialize tests
# ---------------------------------------------------------------------------


class TestTokenCostInitialize:
    @pytest.mark.asyncio
    async def test_initialize_without_cost(self):
        tc = TokenCost(include_cost=False)
        await tc.initialize()
        assert tc._initialized is True
        assert tc._pricing_data is None

    @pytest.mark.asyncio
    async def test_initialize_with_cost_loads_pricing(self):
        tc = TokenCost(include_cost=True)

        with patch.object(TokenCost, "_load_pricing_data", new_callable=AsyncMock) as mock_load:
            await tc.initialize()
            mock_load.assert_called_once()
            assert tc._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        tc = TokenCost(include_cost=False)
        await tc.initialize()
        await tc.initialize()  # Should not fail
        assert tc._initialized is True


# ---------------------------------------------------------------------------
# TokenCost.add_usage tests
# ---------------------------------------------------------------------------


class TestTokenCostAddUsage:
    def test_add_usage_returns_entry(self):
        tc = TokenCost()
        usage = _make_usage()
        entry = tc.add_usage("gpt-4", usage)
        assert entry.model == "gpt-4"
        assert entry.usage.prompt_tokens == 100
        assert entry.usage.completion_tokens == 50

    def test_add_usage_appends_to_history(self):
        tc = TokenCost()
        tc.add_usage("gpt-4", _make_usage())
        tc.add_usage("gpt-4", _make_usage(prompt_tokens=200))
        assert len(tc.usage_history) == 2


# ---------------------------------------------------------------------------
# TokenCost.calculate_cost tests
# ---------------------------------------------------------------------------


class TestTokenCostCalculateCost:
    @pytest.mark.asyncio
    async def test_returns_none_when_not_include_cost(self):
        tc = TokenCost(include_cost=False)
        result = await tc.calculate_cost("gpt-4", _make_usage())
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_pricing(self):
        tc = TokenCost(include_cost=True)
        tc._initialized = True
        tc._pricing_data = {}
        result = await tc.calculate_cost("unknown-model", _make_usage())
        assert result is None

    @pytest.mark.asyncio
    async def test_calculates_cost_from_custom_pricing(self):
        tc = TokenCost(include_cost=True)
        tc._initialized = True
        tc._pricing_data = {}

        with patch(
            "openbrowser.tokens.service.CUSTOM_MODEL_PRICING",
            {
                "test-model": {
                    "input_cost_per_token": 0.001,
                    "output_cost_per_token": 0.002,
                    "max_tokens": 4096,
                    "max_input_tokens": 4096,
                    "max_output_tokens": 4096,
                    "cache_read_input_token_cost": None,
                    "cache_creation_input_token_cost": None,
                }
            },
        ):
            usage = _make_usage(prompt_tokens=100, completion_tokens=50)
            result = await tc.calculate_cost("test-model", usage)
            assert result is not None
            assert isinstance(result, TokenCostCalculated)
            assert result.new_prompt_tokens == 100
            assert result.completion_tokens == 50

    @pytest.mark.asyncio
    async def test_calculates_with_cached_tokens(self):
        tc = TokenCost(include_cost=True)
        tc._initialized = True
        tc._pricing_data = {}

        with patch(
            "openbrowser.tokens.service.CUSTOM_MODEL_PRICING",
            {
                "cached-model": {
                    "input_cost_per_token": 0.001,
                    "output_cost_per_token": 0.002,
                    "max_tokens": 4096,
                    "max_input_tokens": 4096,
                    "max_output_tokens": 4096,
                    "cache_read_input_token_cost": 0.0005,
                    "cache_creation_input_token_cost": 0.0015,
                }
            },
        ):
            usage = _make_usage(
                prompt_tokens=100,
                completion_tokens=50,
                prompt_cached_tokens=30,
                prompt_cache_creation_tokens=10,
            )
            result = await tc.calculate_cost("cached-model", usage)
            assert result is not None
            assert result.prompt_read_cached_tokens == 30
            assert result.prompt_cached_creation_tokens == 10


# ---------------------------------------------------------------------------
# TokenCost.get_model_pricing tests
# ---------------------------------------------------------------------------


class TestTokenCostGetModelPricing:
    @pytest.mark.asyncio
    async def test_custom_pricing_takes_precedence(self):
        tc = TokenCost(include_cost=True)
        tc._initialized = True
        tc._pricing_data = {"test-model": {"input_cost_per_token": 0.999}}

        with patch(
            "openbrowser.tokens.service.CUSTOM_MODEL_PRICING",
            {
                "test-model": {
                    "input_cost_per_token": 0.001,
                    "output_cost_per_token": 0.002,
                }
            },
        ):
            result = await tc.get_model_pricing("test-model")
            assert result is not None
            assert result.input_cost_per_token == 0.001

    @pytest.mark.asyncio
    async def test_falls_back_to_litellm_data(self):
        tc = TokenCost(include_cost=True)
        tc._initialized = True
        tc._pricing_data = {
            "gpt-4": {
                "input_cost_per_token": 0.03,
                "output_cost_per_token": 0.06,
                "max_tokens": 8192,
            }
        }

        with patch(
            "openbrowser.tokens.service.CUSTOM_MODEL_PRICING", {}
        ), patch(
            "openbrowser.tokens.service.MODEL_TO_LITELLM", {"gpt-4": "gpt-4"}
        ):
            result = await tc.get_model_pricing("gpt-4")
            assert result is not None
            assert result.input_cost_per_token == 0.03

    @pytest.mark.asyncio
    async def test_returns_none_for_unknown_model(self):
        tc = TokenCost(include_cost=True)
        tc._initialized = True
        tc._pricing_data = {}

        with patch("openbrowser.tokens.service.CUSTOM_MODEL_PRICING", {}):
            result = await tc.get_model_pricing("unknown-model-xyz")
            assert result is None


# ---------------------------------------------------------------------------
# TokenCost.get_usage_tokens_for_model tests
# ---------------------------------------------------------------------------


class TestTokenCostGetUsageTokens:
    def test_aggregates_usage_for_model(self):
        tc = TokenCost()
        tc.add_usage("gpt-4", _make_usage(prompt_tokens=100, completion_tokens=50))
        tc.add_usage("gpt-4", _make_usage(prompt_tokens=200, completion_tokens=100))
        tc.add_usage("claude", _make_usage(prompt_tokens=300, completion_tokens=150))

        result = tc.get_usage_tokens_for_model("gpt-4")
        assert result.prompt_tokens == 300  # 100 + 200
        assert result.completion_tokens == 150  # 50 + 100
        assert result.total_tokens == 450

    def test_returns_zeros_for_unknown_model(self):
        tc = TokenCost()
        result = tc.get_usage_tokens_for_model("nonexistent")
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.total_tokens == 0


# ---------------------------------------------------------------------------
# TokenCost.get_usage_summary tests
# ---------------------------------------------------------------------------


class TestTokenCostGetUsageSummary:
    @pytest.mark.asyncio
    async def test_empty_history_returns_zeros(self):
        tc = TokenCost()
        summary = await tc.get_usage_summary()
        assert summary.total_tokens == 0
        assert summary.entry_count == 0
        assert summary.total_cost == 0.0

    @pytest.mark.asyncio
    async def test_filters_by_model(self):
        tc = TokenCost()
        tc.add_usage("gpt-4", _make_usage(prompt_tokens=100, completion_tokens=50))
        tc.add_usage("claude", _make_usage(prompt_tokens=200, completion_tokens=100))

        summary = await tc.get_usage_summary(model="gpt-4")
        assert summary.entry_count == 1
        assert summary.total_prompt_tokens == 100

    @pytest.mark.asyncio
    async def test_filters_by_since(self):
        tc = TokenCost()
        tc.add_usage("gpt-4", _make_usage(prompt_tokens=100, completion_tokens=50))
        # Entry was added now, so filtering since the future should return 0
        future = datetime.now() + timedelta(hours=1)
        summary = await tc.get_usage_summary(since=future)
        assert summary.entry_count == 0

    @pytest.mark.asyncio
    async def test_summary_with_cost_tracking(self):
        tc = TokenCost(include_cost=True)
        tc._initialized = True
        tc._pricing_data = {}

        with patch(
            "openbrowser.tokens.service.CUSTOM_MODEL_PRICING",
            {
                "gpt-4": {
                    "input_cost_per_token": 0.001,
                    "output_cost_per_token": 0.002,
                }
            },
        ):
            tc.add_usage("gpt-4", _make_usage(prompt_tokens=100, completion_tokens=50))
            summary = await tc.get_usage_summary()
            assert summary.entry_count == 1
            assert summary.total_tokens == 150
            # Cost should be calculated
            assert summary.total_cost > 0


# ---------------------------------------------------------------------------
# TokenCost._format_tokens tests
# ---------------------------------------------------------------------------


class TestTokenCostFormatTokens:
    def test_small_number(self):
        tc = TokenCost()
        assert tc._format_tokens(42) == "42"

    def test_thousands(self):
        tc = TokenCost()
        result = tc._format_tokens(1500)
        assert result == "1.5k"

    def test_millions(self):
        tc = TokenCost()
        result = tc._format_tokens(2500000)
        assert result == "2.5M"

    def test_billions(self):
        tc = TokenCost()
        result = tc._format_tokens(3000000000)
        assert result == "3.0B"

    def test_exactly_1000(self):
        tc = TokenCost()
        assert tc._format_tokens(1000) == "1.0k"


# ---------------------------------------------------------------------------
# TokenCost.register_llm tests
# ---------------------------------------------------------------------------


class TestTokenCostRegisterLlm:
    def test_register_llm_wraps_ainvoke(self):
        tc = TokenCost()
        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_llm.provider = "test-provider"
        original_ainvoke = AsyncMock()
        mock_llm.ainvoke = original_ainvoke

        result = tc.register_llm(mock_llm)
        assert result is mock_llm
        # ainvoke should have been replaced
        assert mock_llm.ainvoke is not original_ainvoke

    def test_register_same_instance_twice_is_noop(self):
        tc = TokenCost()
        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_llm.provider = "test-provider"
        mock_llm.ainvoke = AsyncMock()

        tc.register_llm(mock_llm)
        first_ainvoke = mock_llm.ainvoke

        tc.register_llm(mock_llm)
        # Should not wrap again
        assert mock_llm.ainvoke is first_ainvoke


# ---------------------------------------------------------------------------
# TokenCost.clear_history tests
# ---------------------------------------------------------------------------


class TestTokenCostClearHistory:
    def test_clear_removes_all_entries(self):
        tc = TokenCost()
        tc.add_usage("gpt-4", _make_usage())
        tc.add_usage("gpt-4", _make_usage())
        assert len(tc.usage_history) == 2

        tc.clear_history()
        assert len(tc.usage_history) == 0


# ---------------------------------------------------------------------------
# TokenCost.refresh_pricing_data tests
# ---------------------------------------------------------------------------


class TestTokenCostRefreshPricing:
    @pytest.mark.asyncio
    async def test_refresh_when_include_cost(self):
        tc = TokenCost(include_cost=True)
        with patch.object(
            TokenCost, "_fetch_and_cache_pricing_data", new_callable=AsyncMock
        ) as mock_fetch:
            await tc.refresh_pricing_data()
            mock_fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_refresh_when_not_include_cost(self):
        tc = TokenCost(include_cost=False)
        with patch.object(
            TokenCost, "_fetch_and_cache_pricing_data", new_callable=AsyncMock
        ) as mock_fetch:
            await tc.refresh_pricing_data()
            mock_fetch.assert_not_called()


# ---------------------------------------------------------------------------
# TokenCost.clean_old_caches tests
# ---------------------------------------------------------------------------


class TestTokenCostCleanOldCaches:
    @pytest.mark.asyncio
    async def test_clean_old_caches_removes_oldest(self, tmp_path):
        tc = TokenCost()
        tc._cache_dir = tmp_path

        # Create some fake cache files
        for i in range(5):
            f = tmp_path / f"pricing_{i}.json"
            f.write_text("{}")

        await tc.clean_old_caches(keep_count=2)
        remaining = list(tmp_path.glob("*.json"))
        assert len(remaining) == 2

    @pytest.mark.asyncio
    async def test_clean_old_caches_noop_when_few_files(self, tmp_path):
        tc = TokenCost()
        tc._cache_dir = tmp_path

        f = tmp_path / "pricing_0.json"
        f.write_text("{}")

        await tc.clean_old_caches(keep_count=3)
        remaining = list(tmp_path.glob("*.json"))
        assert len(remaining) == 1


# ---------------------------------------------------------------------------
# TokenCost._build_input_tokens_display tests
# ---------------------------------------------------------------------------


class TestBuildInputTokensDisplay:
    def test_simple_display_no_cache(self):
        tc = TokenCost()
        usage = _make_usage(prompt_tokens=1000, completion_tokens=500)
        display = tc._build_input_tokens_display(usage, None)
        assert "1.0k" in display

    def test_display_with_cache(self):
        tc = TokenCost()
        usage = _make_usage(
            prompt_tokens=1000,
            completion_tokens=500,
            prompt_cached_tokens=300,
        )
        display = tc._build_input_tokens_display(usage, None)
        # Should show new tokens and cached tokens
        assert "700" in display or "0.7k" in display  # 1000 - 300 = 700 new tokens

    def test_display_with_cost(self):
        tc = TokenCost(include_cost=True)
        usage = _make_usage(prompt_tokens=1000, completion_tokens=500)
        cost = TokenCostCalculated(
            new_prompt_tokens=1000,
            new_prompt_cost=0.05,
            prompt_read_cached_tokens=None,
            prompt_read_cached_cost=None,
            prompt_cached_creation_tokens=None,
            prompt_cache_creation_cost=None,
            completion_tokens=500,
            completion_cost=0.10,
        )
        display = tc._build_input_tokens_display(usage, cost)
        assert "$" in display


# ---------------------------------------------------------------------------
# TokenCost.ensure_pricing_loaded tests
# ---------------------------------------------------------------------------


class TestEnsurePricingLoaded:
    @pytest.mark.asyncio
    async def test_calls_initialize_when_needed(self):
        tc = TokenCost(include_cost=True)
        with patch.object(TokenCost, "initialize", new_callable=AsyncMock) as mock_init:
            await tc.ensure_pricing_loaded()
            mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_when_already_initialized(self):
        tc = TokenCost(include_cost=True)
        tc._initialized = True
        with patch.object(TokenCost, "initialize", new_callable=AsyncMock) as mock_init:
            await tc.ensure_pricing_loaded()
            mock_init.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_cost_not_included(self):
        tc = TokenCost(include_cost=False)
        with patch.object(TokenCost, "initialize", new_callable=AsyncMock) as mock_init:
            await tc.ensure_pricing_loaded()
            mock_init.assert_not_called()
