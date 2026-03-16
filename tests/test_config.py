"""Tests for config.py helper functions."""

import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from openbrowser.config import (
    _get_env_bool_cached,
    create_default_config,
    get_default_llm,
    get_default_profile,
    load_and_migrate_config,
    load_openbrowser_config,
)

logger = logging.getLogger(__name__)


class TestConfig:
    """Tests for config.py helper functions."""

    def test_get_default_profile_extracts_browser_profile(self):
        """get_default_profile returns browser_profile from config dict."""
        config = {
            "browser_profile": {"headless": True, "user_data_dir": "/tmp/test"},
            "llm": {},
            "agent": {},
        }
        result = get_default_profile(config)
        assert result["headless"] is True
        assert result["user_data_dir"] == "/tmp/test"

    def test_get_default_profile_missing_key(self):
        """get_default_profile returns empty dict when key is missing."""
        result = get_default_profile({})
        assert result == {}

    def test_get_default_llm(self):
        """get_default_llm extracts LLM config."""
        config = {"llm": {"model": "gpt-4", "api_key": "test"}}
        result = get_default_llm(config)
        assert result["model"] == "gpt-4"

    def test_get_default_llm_missing(self):
        """get_default_llm returns empty dict when key missing."""
        result = get_default_llm({})
        assert result == {}

    def test_load_openbrowser_config_returns_dict(self):
        """load_openbrowser_config returns a dict with expected keys."""
        # Use a temp config dir to avoid side effects
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with patch.dict(os.environ, {"OPENBROWSER_CONFIG_PATH": str(config_path)}):
                # Clear caches to pick up the new env var
                from openbrowser.config import (
                    _config_cache,
                    _get_env_bool_cached,
                    _get_env_cached,
                    _get_path_cached,
                )

                _config_cache.clear()
                _get_env_cached.cache_clear()
                _get_env_bool_cached.cache_clear()
                _get_path_cached.cache_clear()

                result = load_openbrowser_config()
                assert isinstance(result, dict)
                assert "browser_profile" in result
                assert "llm" in result
                assert "agent" in result

    def test_env_bool_cached_true_values(self):
        """_get_env_bool_cached recognizes true/yes/1."""
        _get_env_bool_cached.cache_clear()

        with patch.dict(os.environ, {"TEST_BOOL_T": "true"}):
            _get_env_bool_cached.cache_clear()
            assert _get_env_bool_cached("TEST_BOOL_T", False) is True

        with patch.dict(os.environ, {"TEST_BOOL_Y": "yes"}):
            _get_env_bool_cached.cache_clear()
            assert _get_env_bool_cached("TEST_BOOL_Y", False) is True

        with patch.dict(os.environ, {"TEST_BOOL_1": "1"}):
            _get_env_bool_cached.cache_clear()
            assert _get_env_bool_cached("TEST_BOOL_1", False) is True

    def test_env_bool_cached_false_values(self):
        """_get_env_bool_cached returns False for false/no/0."""
        _get_env_bool_cached.cache_clear()

        with patch.dict(os.environ, {"TEST_BOOL_F": "false"}):
            _get_env_bool_cached.cache_clear()
            assert _get_env_bool_cached("TEST_BOOL_F", True) is False

    def test_create_default_config(self):
        """create_default_config returns a valid DBStyleConfigJSON."""
        config = create_default_config()
        assert len(config.browser_profile) == 1
        assert len(config.llm) == 1
        assert len(config.agent) == 1

        # Default profile should be marked default=True
        for profile in config.browser_profile.values():
            assert profile.default is True

    def test_load_and_migrate_config_creates_fresh(self):
        """load_and_migrate_config creates fresh config if file missing."""
        from openbrowser.config import _config_cache

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            _config_cache.clear()
            result = load_and_migrate_config(config_path)
            assert config_path.exists()
            assert len(result.browser_profile) > 0
