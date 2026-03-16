"""Deep coverage tests for openbrowser.tokens.service module.

Targets missed lines: 102-107, 111-137, 141-151, 155-162, 166-187, 193,
276-299, 317, 324, 329-333, 376-383, 488-563, 599-602
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openbrowser.llm.views import ChatInvokeCompletion, ChatInvokeUsage
from openbrowser.tokens.service import TokenCost, xdg_cache_home
from openbrowser.tokens.views import (
    CachedPricingData,
    ModelPricing,
    TokenCostCalculated,
    TokenUsageEntry,
)

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
# _load_pricing_data (lines 99-107)
# ---------------------------------------------------------------------------


class TestLoadPricingData:
    """Cover _load_pricing_data: calls _find_valid_cache and routes to either
    _load_from_cache or _fetch_and_cache_pricing_data."""

    @pytest.mark.asyncio
    async def test_load_pricing_data_with_valid_cache(self, tmp_path):
        """Lines 102-105: valid cache file found -> loads from cache."""
        fake_cache_file = tmp_path / "pricing_test.json"

        with patch.object(
            TokenCost, "_find_valid_cache", new_callable=AsyncMock, return_value=fake_cache_file
        ) as mock_find, patch.object(
            TokenCost, "_load_from_cache", new_callable=AsyncMock
        ) as mock_load:
            tc = TokenCost(include_cost=True)
            tc._cache_dir = tmp_path
            await tc._load_pricing_data()
            mock_find.assert_called_once()
            mock_load.assert_called_once_with(fake_cache_file)

    @pytest.mark.asyncio
    async def test_load_pricing_data_no_cache(self, tmp_path):
        """Lines 106-107: no valid cache -> fetch and cache."""
        with patch.object(
            TokenCost, "_find_valid_cache", new_callable=AsyncMock, return_value=None
        ) as mock_find, patch.object(
            TokenCost, "_fetch_and_cache_pricing_data", new_callable=AsyncMock
        ) as mock_fetch:
            tc = TokenCost(include_cost=True)
            tc._cache_dir = tmp_path
            await tc._load_pricing_data()
            mock_find.assert_called_once()
            mock_fetch.assert_called_once()


# ---------------------------------------------------------------------------
# _find_valid_cache (lines 109-137)
# ---------------------------------------------------------------------------


class TestFindValidCache:
    """Cover _find_valid_cache: directory creation, listing, validation, cleanup."""

    @pytest.mark.asyncio
    async def test_find_valid_cache_empty_dir(self, tmp_path):
        """Lines 118-119: no JSON files returns None."""
        tc = TokenCost(include_cost=True)
        tc._cache_dir = tmp_path
        result = await tc._find_valid_cache()
        assert result is None

    @pytest.mark.asyncio
    async def test_find_valid_cache_returns_valid_file(self, tmp_path):
        """Lines 125-127: valid cache file is returned."""
        # Create a valid cache file
        cached = CachedPricingData(timestamp=datetime.now(), data={"test": "data"})
        cache_file = tmp_path / "pricing_valid.json"
        cache_file.write_text(cached.model_dump_json())

        with patch.object(TokenCost, "_is_cache_valid", new_callable=AsyncMock, return_value=True):
            tc = TokenCost(include_cost=True)
            tc._cache_dir = tmp_path
            result = await tc._find_valid_cache()
            assert result is not None
            assert result.name == "pricing_valid.json"

    @pytest.mark.asyncio
    async def test_find_valid_cache_removes_expired(self, tmp_path):
        """Lines 128-133: expired cache files are deleted."""
        expired_file = tmp_path / "pricing_old.json"
        expired_file.write_text("{}")

        with patch.object(TokenCost, "_is_cache_valid", new_callable=AsyncMock, return_value=False):
            tc = TokenCost(include_cost=True)
            tc._cache_dir = tmp_path
            result = await tc._find_valid_cache()
            assert result is None
            # Expired file should be removed
            assert not expired_file.exists()

    @pytest.mark.asyncio
    async def test_find_valid_cache_removes_expired_handles_error(self, tmp_path):
        """Lines 130-133: os.remove raises but is suppressed."""
        expired_file = tmp_path / "pricing_locked.json"
        expired_file.write_text("{}")

        with patch.object(
            TokenCost, "_is_cache_valid", new_callable=AsyncMock, return_value=False
        ), patch("os.remove", side_effect=PermissionError("locked")):
            tc = TokenCost(include_cost=True)
            tc._cache_dir = tmp_path
            result = await tc._find_valid_cache()
            assert result is None

    @pytest.mark.asyncio
    async def test_find_valid_cache_exception_returns_none(self, tmp_path):
        """Lines 136-137: any exception returns None."""
        tc = TokenCost(include_cost=True)
        tc._cache_dir = Path("/nonexistent/impossible/path")

        # Force mkdir to fail
        with patch.object(Path, "mkdir", side_effect=PermissionError("nope")):
            result = await tc._find_valid_cache()
            assert result is None


# ---------------------------------------------------------------------------
# _is_cache_valid (lines 139-151)
# ---------------------------------------------------------------------------


class TestIsCacheValid:
    """Cover _is_cache_valid: file existence, content parsing, expiration check."""

    @pytest.mark.asyncio
    async def test_cache_valid_nonexistent_file(self, tmp_path):
        """Lines 142-143: non-existent file returns False."""
        tc = TokenCost(include_cost=True)
        result = await tc._is_cache_valid(tmp_path / "nonexistent.json")
        assert result is False

    @pytest.mark.asyncio
    async def test_cache_valid_fresh_file(self, tmp_path):
        """Lines 146-149: fresh valid cache file returns True."""
        tc = TokenCost(include_cost=True)
        cached = CachedPricingData(timestamp=datetime.now(), data={"gpt-4": {}})
        cache_file = tmp_path / "fresh.json"
        cache_file.write_text(cached.model_dump_json())
        result = await tc._is_cache_valid(cache_file)
        assert result is True

    @pytest.mark.asyncio
    async def test_cache_valid_expired_file(self, tmp_path):
        """Lines 148-149: expired cache returns False."""
        tc = TokenCost(include_cost=True)
        # Create a cache file with old timestamp
        old_time = datetime.now() - timedelta(days=2)
        cached = CachedPricingData(timestamp=old_time, data={"gpt-4": {}})
        cache_file = tmp_path / "expired.json"
        cache_file.write_text(cached.model_dump_json())
        result = await tc._is_cache_valid(cache_file)
        assert result is False

    @pytest.mark.asyncio
    async def test_cache_valid_corrupt_file(self, tmp_path):
        """Lines 150-151: corrupt file returns False (exception caught)."""
        tc = TokenCost(include_cost=True)
        cache_file = tmp_path / "corrupt.json"
        cache_file.write_text("not valid json {{{")
        result = await tc._is_cache_valid(cache_file)
        assert result is False


# ---------------------------------------------------------------------------
# _load_from_cache (lines 153-162)
# ---------------------------------------------------------------------------


class TestLoadFromCache:
    """Cover _load_from_cache: read and parse, fallback on error."""

    @pytest.mark.asyncio
    async def test_load_from_cache_success(self, tmp_path):
        """Lines 155-158: successfully loads pricing data from cache."""
        tc = TokenCost(include_cost=True)
        cached = CachedPricingData(timestamp=datetime.now(), data={"gpt-4": {"input_cost_per_token": 0.03}})
        cache_file = tmp_path / "valid.json"
        cache_file.write_text(cached.model_dump_json())

        await tc._load_from_cache(cache_file)
        assert tc._pricing_data is not None
        assert "gpt-4" in tc._pricing_data

    @pytest.mark.asyncio
    async def test_load_from_cache_error_falls_back(self, tmp_path):
        """Lines 159-162: error reading cache falls back to fetching."""
        cache_file = tmp_path / "bad.json"
        cache_file.write_text("not valid")

        with patch.object(TokenCost, "_fetch_and_cache_pricing_data", new_callable=AsyncMock) as mock_fetch:
            tc = TokenCost(include_cost=True)
            await tc._load_from_cache(cache_file)
            mock_fetch.assert_called_once()


# ---------------------------------------------------------------------------
# _fetch_and_cache_pricing_data (lines 164-187)
# ---------------------------------------------------------------------------


class TestFetchAndCachePricingData:
    """Cover _fetch_and_cache_pricing_data: HTTP fetch, write cache, error handling."""

    @pytest.mark.asyncio
    async def test_fetch_success(self, tmp_path):
        """Lines 166-183: successful fetch and cache write."""
        tc = TokenCost(include_cost=True)
        tc._cache_dir = tmp_path

        mock_response = MagicMock()
        mock_response.json.return_value = {"gpt-4": {"input_cost_per_token": 0.03}}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("openbrowser.tokens.service.httpx.AsyncClient", return_value=mock_client):
            await tc._fetch_and_cache_pricing_data()

        assert tc._pricing_data is not None
        assert "gpt-4" in tc._pricing_data
        # Cache file should have been written
        cache_files = list(tmp_path.glob("pricing_*.json"))
        assert len(cache_files) == 1

    @pytest.mark.asyncio
    async def test_fetch_failure(self, tmp_path):
        """Lines 184-187: network error falls back to empty dict."""
        tc = TokenCost(include_cost=True)
        tc._cache_dir = tmp_path

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("network error"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("openbrowser.tokens.service.httpx.AsyncClient", return_value=mock_client):
            await tc._fetch_and_cache_pricing_data()

        assert tc._pricing_data == {}


# ---------------------------------------------------------------------------
# get_model_pricing - uninitialized path (line 193)
# ---------------------------------------------------------------------------


class TestGetModelPricingAutoInit:
    """Cover line 193: auto-initialization when not yet initialized."""

    @pytest.mark.asyncio
    async def test_auto_init_when_not_initialized(self):
        """Line 193: calls initialize when _initialized is False."""
        with patch.object(TokenCost, "initialize", new_callable=AsyncMock) as mock_init:
            tc = TokenCost(include_cost=False)
            assert tc._initialized is False
            # After init we still have no pricing data
            result = await tc.get_model_pricing("nonexistent")
            mock_init.assert_called_once()


# ---------------------------------------------------------------------------
# _log_usage (lines 274-299)
# ---------------------------------------------------------------------------


class TestLogUsage:
    """Cover _log_usage: logging with/without cost, auto-initialization."""

    @pytest.mark.asyncio
    async def test_log_usage_no_cost(self):
        """Lines 276-299: log usage without cost tracking."""
        with patch.object(TokenCost, "calculate_cost", new_callable=AsyncMock, return_value=None):
            tc = TokenCost(include_cost=False)
            tc._initialized = True
            usage_entry = tc.add_usage("gpt-4", _make_usage(prompt_tokens=1000, completion_tokens=500))
            await tc._log_usage("gpt-4", usage_entry)
        # Should not raise, just log

    @pytest.mark.asyncio
    async def test_log_usage_with_cost(self):
        """Lines 294-299: log usage with cost tracking and cost > 0."""
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
        with patch.object(TokenCost, "calculate_cost", new_callable=AsyncMock, return_value=cost):
            tc = TokenCost(include_cost=True)
            tc._initialized = True
            usage_entry = tc.add_usage("gpt-4", _make_usage(prompt_tokens=1000, completion_tokens=500))
            await tc._log_usage("gpt-4", usage_entry)

    @pytest.mark.asyncio
    async def test_log_usage_auto_init(self):
        """Line 276-277: _log_usage auto-initializes if not initialized."""
        with patch.object(TokenCost, "initialize", new_callable=AsyncMock) as mock_init, \
             patch.object(TokenCost, "calculate_cost", new_callable=AsyncMock, return_value=None):
            tc = TokenCost(include_cost=False)
            tc._initialized = False
            usage_entry = TokenUsageEntry(
                model="gpt-4", timestamp=datetime.now(), usage=_make_usage()
            )
            await tc._log_usage("gpt-4", usage_entry)
            mock_init.assert_called_once()


# ---------------------------------------------------------------------------
# _build_input_tokens_display - cost branches (lines 316-333)
# ---------------------------------------------------------------------------


class TestBuildInputTokensDisplayDeep:
    """Cover cost display branches: lines 317, 324, 329-333."""

    def test_display_with_cost_and_new_tokens(self):
        """Line 317: new tokens with cost > 0 shows dollar amount."""
        tc = TokenCost(include_cost=True)
        usage = _make_usage(prompt_tokens=1000, completion_tokens=500, prompt_cached_tokens=300)
        cost = TokenCostCalculated(
            new_prompt_tokens=700,
            new_prompt_cost=0.07,
            prompt_read_cached_tokens=300,
            prompt_read_cached_cost=None,
            prompt_cached_creation_tokens=None,
            prompt_cache_creation_cost=None,
            completion_tokens=500,
            completion_cost=0.10,
        )
        display = tc._build_input_tokens_display(usage, cost)
        assert "$0.07" in display

    def test_display_with_cached_cost(self):
        """Line 324: cached tokens with cost shows dollar amount."""
        tc = TokenCost(include_cost=True)
        usage = _make_usage(prompt_tokens=1000, completion_tokens=500, prompt_cached_tokens=300)
        cost = TokenCostCalculated(
            new_prompt_tokens=700,
            new_prompt_cost=0.07,
            prompt_read_cached_tokens=300,
            prompt_read_cached_cost=0.015,
            prompt_cached_creation_tokens=None,
            prompt_cache_creation_cost=None,
            completion_tokens=500,
            completion_cost=0.10,
        )
        display = tc._build_input_tokens_display(usage, cost)
        assert "$0.0150" in display

    def test_display_with_cache_creation_cost(self):
        """Lines 329-331: cache creation tokens with cost."""
        tc = TokenCost(include_cost=True)
        usage = _make_usage(
            prompt_tokens=1000,
            completion_tokens=500,
            prompt_cached_tokens=300,
            prompt_cache_creation_tokens=200,
        )
        cost = TokenCostCalculated(
            new_prompt_tokens=500,
            new_prompt_cost=0.05,
            prompt_read_cached_tokens=300,
            prompt_read_cached_cost=0.015,
            prompt_cached_creation_tokens=200,
            prompt_cache_creation_cost=0.03,
            completion_tokens=500,
            completion_cost=0.10,
        )
        display = tc._build_input_tokens_display(usage, cost)
        assert "$0.0300" in display

    def test_display_cache_creation_without_cost(self):
        """Lines 332-333: cache creation tokens without cost data."""
        tc = TokenCost(include_cost=False)
        usage = _make_usage(
            prompt_tokens=1000,
            completion_tokens=500,
            prompt_cached_tokens=300,
            prompt_cache_creation_tokens=200,
        )
        display = tc._build_input_tokens_display(usage, None)
        # Should show token counts without dollar amounts
        assert "$" not in display
        assert "200" in display

    def test_display_fallback_with_cost(self):
        """Lines 338-339: fallback (no cache info) with cost > 0."""
        tc = TokenCost(include_cost=True)
        usage = _make_usage(prompt_tokens=1000, completion_tokens=500, prompt_cached_tokens=0)
        cost = TokenCostCalculated(
            new_prompt_tokens=1000,
            new_prompt_cost=0.10,
            prompt_read_cached_tokens=None,
            prompt_read_cached_cost=None,
            prompt_cached_creation_tokens=None,
            prompt_cache_creation_cost=None,
            completion_tokens=500,
            completion_cost=0.10,
        )
        display = tc._build_input_tokens_display(usage, cost)
        assert "$0.10" in display


# ---------------------------------------------------------------------------
# register_llm - tracked_ainvoke (lines 367-383)
# ---------------------------------------------------------------------------


class TestRegisterLlmTrackedAinvoke:
    """Cover the tracked_ainvoke closure: lines 376-383."""

    @pytest.mark.asyncio
    async def test_tracked_ainvoke_records_usage(self):
        """Lines 373-378: tracked ainvoke calls add_usage and creates log task."""
        with patch.object(TokenCost, "_log_usage", new_callable=AsyncMock):
            tc = TokenCost()
            mock_llm = MagicMock()
            mock_llm.model = "test-model"
            mock_llm.provider = "test-provider"

            mock_usage = _make_usage(prompt_tokens=100, completion_tokens=50)
            mock_result = MagicMock()
            mock_result.usage = mock_usage
            original_ainvoke = AsyncMock(return_value=mock_result)
            mock_llm.ainvoke = original_ainvoke

            tc.register_llm(mock_llm)

            # Call the wrapped ainvoke
            result = await mock_llm.ainvoke([], None)
            assert result is mock_result
            assert len(tc.usage_history) == 1
            assert tc.usage_history[0].model == "test-model"

    @pytest.mark.asyncio
    async def test_tracked_ainvoke_no_usage(self):
        """Lines 376-383: tracked ainvoke with no usage skips tracking."""
        tc = TokenCost()
        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_llm.provider = "test-provider"

        mock_result = MagicMock()
        mock_result.usage = None
        original_ainvoke = AsyncMock(return_value=mock_result)
        mock_llm.ainvoke = original_ainvoke

        tc.register_llm(mock_llm)

        result = await mock_llm.ainvoke([], None)
        assert result is mock_result
        # No usage tracked
        assert len(tc.usage_history) == 0

    @pytest.mark.asyncio
    async def test_tracked_ainvoke_with_kwargs(self):
        """Tracked ainvoke passes through additional kwargs."""
        tc = TokenCost()
        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_llm.provider = "test-provider"

        mock_result = MagicMock()
        mock_result.usage = None
        original_ainvoke = AsyncMock(return_value=mock_result)
        mock_llm.ainvoke = original_ainvoke

        tc.register_llm(mock_llm)

        result = await mock_llm.ainvoke([], None, temperature=0.5)
        original_ainvoke.assert_called_once_with([], None, temperature=0.5)


# ---------------------------------------------------------------------------
# log_usage_summary (lines 486-567)
# ---------------------------------------------------------------------------


class TestLogUsageSummary:
    """Cover log_usage_summary: empty history, single model, multi-model,
    with/without cost."""

    @pytest.mark.asyncio
    async def test_log_usage_summary_empty(self):
        """Lines 488-489: no usage history returns early."""
        tc = TokenCost()
        # Should not raise
        await tc.log_usage_summary()

    @pytest.mark.asyncio
    async def test_log_usage_summary_zero_entries(self):
        """Lines 493-494: summary with zero entries returns early."""
        from openbrowser.tokens.views import UsageSummary

        empty_summary = UsageSummary(
            total_prompt_tokens=0,
            total_prompt_cost=0.0,
            total_prompt_cached_tokens=0,
            total_prompt_cached_cost=0.0,
            total_completion_tokens=0,
            total_completion_cost=0.0,
            total_tokens=0,
            total_cost=0.0,
            entry_count=0,
        )

        with patch.object(
            TokenCost, "get_usage_summary", new_callable=AsyncMock, return_value=empty_summary
        ):
            tc = TokenCost()
            tc.usage_history = [MagicMock()]  # Non-empty to pass first guard
            await tc.log_usage_summary()

    @pytest.mark.asyncio
    async def test_log_usage_summary_single_model_no_cost(self):
        """Lines 506-567: single model without cost tracking."""
        tc = TokenCost(include_cost=False)
        tc._initialized = True
        tc.add_usage("gpt-4", _make_usage(prompt_tokens=1000, completion_tokens=500))

        await tc.log_usage_summary()

    @pytest.mark.asyncio
    async def test_log_usage_summary_multi_model_no_cost(self):
        """Lines 520-524: multi-model total summary line."""
        tc = TokenCost(include_cost=False)
        tc._initialized = True
        tc.add_usage("gpt-4", _make_usage(prompt_tokens=1000, completion_tokens=500))
        tc.add_usage("claude", _make_usage(prompt_tokens=2000, completion_tokens=1000))

        await tc.log_usage_summary()

    @pytest.mark.asyncio
    async def test_log_usage_summary_with_cost(self):
        """Lines 511-551: log usage summary with cost tracking enabled."""
        tc = TokenCost(include_cost=True)
        tc._initialized = True
        tc._pricing_data = {}

        with patch(
            "openbrowser.tokens.service.CUSTOM_MODEL_PRICING",
            {
                "gpt-4": {
                    "input_cost_per_token": 0.001,
                    "output_cost_per_token": 0.002,
                },
                "claude": {
                    "input_cost_per_token": 0.0005,
                    "output_cost_per_token": 0.001,
                },
            },
        ):
            tc.add_usage("gpt-4", _make_usage(prompt_tokens=1000, completion_tokens=500))
            tc.add_usage("claude", _make_usage(prompt_tokens=2000, completion_tokens=1000))
            await tc.log_usage_summary()

    @pytest.mark.asyncio
    async def test_log_usage_summary_with_cost_zero(self):
        """Lines 554-557: cost tracking on but model cost is 0."""
        tc = TokenCost(include_cost=True)
        tc._initialized = True
        tc._pricing_data = {}

        # Model with zero cost
        with patch(
            "openbrowser.tokens.service.CUSTOM_MODEL_PRICING",
            {
                "free-model": {
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                },
            },
        ):
            tc.add_usage("free-model", _make_usage(prompt_tokens=1000, completion_tokens=500))
            await tc.log_usage_summary()


# ---------------------------------------------------------------------------
# clean_old_caches - error handling (lines 597-602)
# ---------------------------------------------------------------------------


class TestCleanOldCachesDeep:
    """Cover clean_old_caches error paths: lines 599-602."""

    @pytest.mark.asyncio
    async def test_clean_old_caches_os_remove_error(self, tmp_path):
        """Lines 599-600: os.remove fails silently."""
        tc = TokenCost()
        tc._cache_dir = tmp_path

        for i in range(5):
            f = tmp_path / f"pricing_{i}.json"
            f.write_text("{}")
            # Stagger modification times
            os.utime(f, (time.time() - (5 - i), time.time() - (5 - i)))

        with patch("os.remove", side_effect=PermissionError("locked")):
            await tc.clean_old_caches(keep_count=2)
            # Should not raise

    @pytest.mark.asyncio
    async def test_clean_old_caches_glob_error(self):
        """Lines 601-602: exception in glob returns gracefully."""
        tc = TokenCost()
        tc._cache_dir = Path("/nonexistent/path")

        await tc.clean_old_caches(keep_count=2)
        # Should not raise


# ---------------------------------------------------------------------------
# get_cost_by_model
# ---------------------------------------------------------------------------


class TestGetCostByModel:
    """Cover get_cost_by_model (lines 569-572)."""

    @pytest.mark.asyncio
    async def test_get_cost_by_model(self):
        tc = TokenCost()
        tc.add_usage("gpt-4", _make_usage())
        result = await tc.get_cost_by_model()
        assert "gpt-4" in result
        assert result["gpt-4"].invocations == 1


# ---------------------------------------------------------------------------
# get_usage_summary with cost (deeper branches)
# ---------------------------------------------------------------------------


class TestGetUsageSummaryDeep:
    """Cover deeper branches in get_usage_summary (lines 449-461)."""

    @pytest.mark.asyncio
    async def test_summary_cost_calculation_per_entry(self):
        """Lines 449-461: cost calculated per entry and averages computed."""
        tc = TokenCost(include_cost=True)
        tc._initialized = True
        tc._pricing_data = {}

        with patch(
            "openbrowser.tokens.service.CUSTOM_MODEL_PRICING",
            {
                "gpt-4": {
                    "input_cost_per_token": 0.001,
                    "output_cost_per_token": 0.002,
                    "cache_read_input_token_cost": 0.0005,
                    "cache_creation_input_token_cost": None,
                },
            },
        ):
            tc.add_usage("gpt-4", _make_usage(prompt_tokens=100, completion_tokens=50, prompt_cached_tokens=30))
            tc.add_usage("gpt-4", _make_usage(prompt_tokens=200, completion_tokens=100, prompt_cached_tokens=50))
            summary = await tc.get_usage_summary()

            assert summary.entry_count == 2
            assert summary.total_cost > 0
            assert "gpt-4" in summary.by_model
            stats = summary.by_model["gpt-4"]
            assert stats.invocations == 2
            assert stats.average_tokens_per_invocation > 0


# ---------------------------------------------------------------------------
# xdg_cache_home with custom path
# ---------------------------------------------------------------------------


class TestXdgCacheHomeDeep:
    """Cover xdg_cache_home with custom XDG_CACHE_HOME set."""

    def test_custom_xdg_cache_home(self, tmp_path):
        """Line 55-56: custom XDG_CACHE_HOME is used when absolute."""
        xdg_cache_home.cache_clear()
        with patch("openbrowser.tokens.service.CONFIG") as mock_config:
            mock_config.XDG_CACHE_HOME = str(tmp_path)
            xdg_cache_home.cache_clear()
            result = xdg_cache_home()
            assert result == tmp_path

    def test_relative_xdg_cache_home_falls_back(self):
        """Line 55: relative XDG_CACHE_HOME falls back to default."""
        xdg_cache_home.cache_clear()
        with patch("openbrowser.tokens.service.CONFIG") as mock_config:
            mock_config.XDG_CACHE_HOME = "relative/path"
            xdg_cache_home.cache_clear()
            result = xdg_cache_home()
            assert result == Path.home() / ".cache"
