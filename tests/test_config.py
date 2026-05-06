"""Tests for config.py helper functions."""

import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from openbrowser.config import (
    CONFIG,
    _get_env_bool_cached,
    _get_env_cached,
    _get_path_cached,
    _config_cache,
    apply_managed_browser_profile_defaults,
    create_default_config,
    get_default_llm,
    get_default_profile,
    is_openbrowser_managed_profile_dir,
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
                _config_cache.clear()
                CONFIG._env_config = None
                _get_env_cached.cache_clear()
                _get_env_bool_cached.cache_clear()
                _get_path_cached.cache_clear()

                result = load_openbrowser_config()
                assert isinstance(result, dict)
                assert "browser_profile" in result
                assert "llm" in result
                assert "agent" in result

    def test_load_openbrowser_config_applies_user_data_env_overrides(self):
        """OPENBROWSER_USER_DATA_DIR and OPENBROWSER_STORAGE_STATE override browser profile config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            env = {
                "OPENBROWSER_CONFIG_PATH": str(config_path),
                "OPENBROWSER_USER_DATA_DIR": "/tmp/custom-profile",
                "OPENBROWSER_STORAGE_STATE": "/tmp/custom-profile/state.json",
            }
            with patch.dict(os.environ, env, clear=False):
                _config_cache.clear()
                CONFIG._env_config = None
                _get_env_cached.cache_clear()
                _get_env_bool_cached.cache_clear()
                _get_path_cached.cache_clear()

                result = load_openbrowser_config()
                assert result["browser_profile"]["user_data_dir"] == "/tmp/custom-profile"
                assert result["browser_profile"]["storage_state"] == "/tmp/custom-profile/state.json"

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

    def test_apply_managed_browser_profile_defaults_adds_storage_state(self):
        """Managed OpenBrowser profile dirs get a sibling storage_state.json path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "openbrowser-config"
            env = {"OPENBROWSER_CONFIG_DIR": str(config_dir)}
            with patch.dict(os.environ, env, clear=False):
                _config_cache.clear()
                CONFIG._env_config = None
                _get_env_cached.cache_clear()
                _get_env_bool_cached.cache_clear()
                _get_path_cached.cache_clear()

                profile_dir = config_dir / "profiles" / "daemon"
                result = apply_managed_browser_profile_defaults({}, profile_dir)
                assert result["user_data_dir"] == str(profile_dir)
                assert result["storage_state"] == str(profile_dir / "storage_state.json")
                assert is_openbrowser_managed_profile_dir(profile_dir) is True

    def test_apply_managed_browser_profile_defaults_skips_external_storage_state(self):
        """External Chrome profiles are respected without auto-injecting storage_state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "openbrowser-config"
            external_dir = Path(tmpdir) / "external-chrome-profile"
            env = {"OPENBROWSER_CONFIG_DIR": str(config_dir)}
            with patch.dict(os.environ, env, clear=False):
                _config_cache.clear()
                CONFIG._env_config = None
                _get_env_cached.cache_clear()
                _get_env_bool_cached.cache_clear()
                _get_path_cached.cache_clear()

                result = apply_managed_browser_profile_defaults({"user_data_dir": str(external_dir)})
                assert result["user_data_dir"] == str(external_dir)
                assert "storage_state" not in result
                assert is_openbrowser_managed_profile_dir(external_dir) is False

    def test_load_and_migrate_config_creates_fresh(self):
        """load_and_migrate_config creates fresh config if file missing."""
        from openbrowser.config import _config_cache

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            _config_cache.clear()
            result = load_and_migrate_config(config_path)
            assert config_path.exists()
            assert len(result.browser_profile) > 0
