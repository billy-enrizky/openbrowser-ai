"""Comprehensive tests for src/openbrowser/config.py to cover remaining gaps.

Missing lines: 31, 39-40, 47-49, 107, 115, 138, 149, 153, 157, 161, 165, 169, 173,
177, 181, 185, 194, 198, 334-335, 363-384, 434, 436, 438, 441-444, 450, 452,
470-473, 483-486, 496-499, 514, 517-518, 523, 526, 528, 530, 533-534, 540
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

logger = logging.getLogger(__name__)


class TestIsRunningInDocker:
    """Test is_running_in_docker function."""

    def test_dockerenv_exists(self):
        """Line 31: .dockerenv file exists."""
        from openbrowser.config import is_running_in_docker

        is_running_in_docker.cache_clear()
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.read_text", return_value="docker"):
                result = is_running_in_docker()
                assert result is True
        is_running_in_docker.cache_clear()

    def test_proc_cgroup_docker(self):
        """Line 31: /proc/1/cgroup contains docker."""
        from openbrowser.config import is_running_in_docker

        is_running_in_docker.cache_clear()

        call_count = [0]
        original_exists = Path.exists

        def mock_exists(self):
            path_str = str(self)
            if path_str == "/.dockerenv":
                return False
            if path_str == "/proc/1/cgroup":
                return True
            return original_exists(self)

        with patch.object(Path, "exists", mock_exists):
            with patch.object(Path, "read_text", return_value="12:blkio:/docker/abc123"):
                result = is_running_in_docker()
                assert result is True
        is_running_in_docker.cache_clear()

    def test_init_proc_python(self):
        """Lines 39-40: init process looks like python."""
        from openbrowser.config import is_running_in_docker

        is_running_in_docker.cache_clear()

        mock_proc = MagicMock()
        mock_proc.cmdline.return_value = ["python", "-m", "uvicorn"]

        with patch("pathlib.Path.exists", return_value=False):
            with patch("pathlib.Path.read_text", side_effect=Exception("no file")):
                with patch("psutil.Process", return_value=mock_proc):
                    with patch("psutil.pids", return_value=list(range(100))):
                        result = is_running_in_docker()
                        assert result is True
        is_running_in_docker.cache_clear()

    def test_few_processes(self):
        """Lines 47-49: fewer than 10 processes."""
        from openbrowser.config import is_running_in_docker

        is_running_in_docker.cache_clear()

        mock_proc = MagicMock()
        mock_proc.cmdline.return_value = ["bash"]

        with patch("pathlib.Path.exists", return_value=False):
            with patch("pathlib.Path.read_text", side_effect=Exception("no file")):
                with patch("psutil.Process", return_value=mock_proc):
                    with patch("psutil.pids", return_value=[1, 2, 3]):
                        result = is_running_in_docker()
                        assert result is True
        is_running_in_docker.cache_clear()

    def test_not_in_docker(self):
        """Test not in docker."""
        from openbrowser.config import is_running_in_docker

        is_running_in_docker.cache_clear()

        mock_proc = MagicMock()
        mock_proc.cmdline.return_value = ["systemd"]

        with patch("pathlib.Path.exists", return_value=False):
            with patch("pathlib.Path.read_text", side_effect=Exception("no file")):
                with patch("psutil.Process", return_value=mock_proc):
                    with patch("psutil.pids", return_value=list(range(100))):
                        result = is_running_in_docker()
                        assert result is False
        is_running_in_docker.cache_clear()


class TestCachedEnvFunctions:
    """Test cached environment variable functions."""

    def test_get_env_cached(self):
        """Test _get_env_cached."""
        from openbrowser.config import _get_env_cached

        _get_env_cached.cache_clear()
        with patch.dict(os.environ, {"TEST_VAR_12345": "value123"}):
            result = _get_env_cached("TEST_VAR_12345")
            assert result == "value123"
        _get_env_cached.cache_clear()

    def test_get_env_bool_cached_true(self):
        """Test _get_env_bool_cached with true value."""
        from openbrowser.config import _get_env_bool_cached

        _get_env_bool_cached.cache_clear()
        with patch.dict(os.environ, {"TEST_BOOL": "true"}):
            result = _get_env_bool_cached("TEST_BOOL")
            assert result is True
        _get_env_bool_cached.cache_clear()

    def test_get_env_bool_cached_false(self):
        """Test _get_env_bool_cached with false value."""
        from openbrowser.config import _get_env_bool_cached

        _get_env_bool_cached.cache_clear()
        with patch.dict(os.environ, {"TEST_BOOL_F": "false"}):
            result = _get_env_bool_cached("TEST_BOOL_F")
            assert result is False
        _get_env_bool_cached.cache_clear()

    def test_get_env_bool_cached_empty(self):
        """Test _get_env_bool_cached with empty value."""
        from openbrowser.config import _get_env_bool_cached

        _get_env_bool_cached.cache_clear()
        with patch.dict(os.environ, {"TEST_BOOL_EMPTY": ""}):
            result = _get_env_bool_cached("TEST_BOOL_EMPTY", default=True)
            assert result is True
        _get_env_bool_cached.cache_clear()

    def test_get_path_cached(self):
        """Test _get_path_cached."""
        from openbrowser.config import _get_path_cached

        _get_path_cached.cache_clear()
        result = _get_path_cached("NONEXISTENT_PATH_VAR", "/tmp/test")
        assert isinstance(result, Path)
        _get_path_cached.cache_clear()


class TestOldConfig:
    """Test OldConfig class properties."""

    def test_all_properties(self, tmp_path):
        """Lines 107-198: all OldConfig properties."""
        from openbrowser.config import OldConfig, _get_env_cached, _get_env_bool_cached, _get_path_cached

        # Clear caches
        _get_env_cached.cache_clear()
        _get_env_bool_cached.cache_clear()
        _get_path_cached.cache_clear()

        # Redirect config dirs to tmp_path so we never touch real user directories
        OldConfig._dirs_created = False
        with patch.dict(os.environ, {
            "OPENBROWSER_CONFIG_DIR": str(tmp_path / "config"),
            "XDG_CONFIG_HOME": str(tmp_path / "xdg_config"),
            "XDG_CACHE_HOME": str(tmp_path / "xdg_cache"),
        }):
            # Clear caches again after setting env vars
            _get_env_cached.cache_clear()
            _get_env_bool_cached.cache_clear()
            _get_path_cached.cache_clear()

            config = OldConfig()

            # Test each property (covers all @property lines)
            assert isinstance(config.OPENBROWSER_LOGGING_LEVEL, str)  # line 88
            assert isinstance(config.ANONYMIZED_TELEMETRY, bool)  # line 92
            assert isinstance(config.XDG_CACHE_HOME, Path)  # line 97
            assert isinstance(config.XDG_CONFIG_HOME, Path)  # line 101
            assert isinstance(config.OPENBROWSER_CONFIG_DIR, Path)  # line 107
            assert isinstance(config.OPENBROWSER_CONFIG_FILE, Path)  # line 115
            assert isinstance(config.OPENBROWSER_PROFILES_DIR, Path)  # line 119
            assert isinstance(config.OPENBROWSER_DEFAULT_USER_DATA_DIR, Path)  # line 125
            assert isinstance(config.OPENBROWSER_EXTENSIONS_DIR, Path)  # line 129

        # Clear caches and reset after test
        _get_env_cached.cache_clear()
        _get_env_bool_cached.cache_clear()
        _get_path_cached.cache_clear()
        OldConfig._dirs_created = False

        # API key properties
        assert isinstance(config.OPENAI_API_KEY, str)  # line 149
        assert isinstance(config.ANTHROPIC_API_KEY, str)  # line 153
        assert isinstance(config.GOOGLE_API_KEY, str)  # line 157
        assert isinstance(config.DEEPSEEK_API_KEY, str)  # line 161
        assert isinstance(config.GROK_API_KEY, str)  # line 165
        assert isinstance(config.NOVITA_API_KEY, str)  # line 169
        assert isinstance(config.AZURE_OPENAI_ENDPOINT, str)  # line 173
        assert isinstance(config.AZURE_OPENAI_KEY, str)  # line 177
        assert isinstance(config.SKIP_LLM_API_KEY_VERIFICATION, bool)  # line 181
        assert isinstance(config.DEFAULT_LLM, str)  # line 185

        # Runtime hints
        assert isinstance(config.IN_DOCKER, bool)  # line 190
        assert isinstance(config.IS_IN_EVALS, bool)  # line 194
        assert isinstance(config.WIN_FONT_DIR, str)  # line 198

    def test_config_dir_custom(self):
        """Line 107: custom OPENBROWSER_CONFIG_DIR."""
        from openbrowser.config import OldConfig

        OldConfig._dirs_created = False
        with patch.dict(os.environ, {"OPENBROWSER_CONFIG_DIR": "/tmp/test_config"}):
            config = OldConfig()
            config_dir = config.OPENBROWSER_CONFIG_DIR
            assert "/tmp/test_config" in str(config_dir)
        OldConfig._dirs_created = False

    def test_ensure_dirs_custom(self):
        """Line 138: _ensure_dirs with custom config dir."""
        from openbrowser.config import OldConfig

        OldConfig._dirs_created = False
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_dir = os.path.join(tmpdir, "custom_config")
            with patch.dict(os.environ, {"OPENBROWSER_CONFIG_DIR": custom_dir}):
                config = OldConfig()
                config._ensure_dirs()
                assert Path(custom_dir).exists()
                assert (Path(custom_dir) / "profiles").exists()
                assert (Path(custom_dir) / "extensions").exists()
        OldConfig._dirs_created = False


class TestLoadAndMigrateConfig:
    """Test load_and_migrate_config function."""

    def test_cached_config(self):
        """Lines 334-335: config cache hit."""
        from openbrowser.config import load_and_migrate_config, _config_cache

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config1 = load_and_migrate_config(config_path)
            # Second call should hit cache
            config2 = load_and_migrate_config(config_path)
            assert config1 is config2
            # Clear cache
            _config_cache.clear()

    def test_new_config_created(self):
        """Test creating new config when file doesn't exist."""
        from openbrowser.config import load_and_migrate_config, _config_cache

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "subdir" / "config.json"
            config = load_and_migrate_config(config_path)
            assert config_path.exists()
            assert len(config.browser_profile) > 0
            _config_cache.clear()

    def test_old_format_migration(self):
        """Lines 363-384: old format config gets replaced."""
        from openbrowser.config import load_and_migrate_config, _config_cache

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            # Write old format config
            old_config = {
                "browser_profile": {"headless": False},
                "llm": {"model": "gpt-4"},
                "agent": {"max_steps": 10},
            }
            with open(config_path, "w") as f:
                json.dump(old_config, f)

            config = load_and_migrate_config(config_path)
            # Should have created new DB-style config
            assert len(config.browser_profile) > 0
            _config_cache.clear()

    def test_valid_db_style_config(self):
        """Test loading valid DB-style config."""
        from openbrowser.config import (
            load_and_migrate_config,
            create_default_config,
            _config_cache,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            default_config = create_default_config()
            with open(config_path, "w") as f:
                json.dump(default_config.model_dump(), f)

            config = load_and_migrate_config(config_path)
            assert len(config.browser_profile) > 0
            _config_cache.clear()

    def test_corrupt_config_recovery(self):
        """Lines 374-384: corrupt config triggers fresh creation."""
        from openbrowser.config import load_and_migrate_config, _config_cache

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            # Write invalid JSON
            config_path.write_text("NOT VALID JSON {{{")

            config = load_and_migrate_config(config_path)
            assert len(config.browser_profile) > 0
            _config_cache.clear()


class TestConfigClass:
    """Test Config class."""

    def test_getattr_old_config(self):
        """Lines 434-435: __getattr__ proxies to OldConfig."""
        from openbrowser.config import Config

        config = Config()
        # Access a known OldConfig property
        level = config.OPENBROWSER_LOGGING_LEVEL
        assert isinstance(level, str)

    def test_getattr_env_config(self):
        """Lines 436-438: __getattr__ falls through to env config."""
        from openbrowser.config import Config

        config = Config()
        # OPENBROWSER_CONFIG_PATH is only in FlatEnvConfig
        val = config.OPENBROWSER_CONFIG_PATH
        assert val is None or isinstance(val, str)

    def test_getattr_special_methods(self):
        """Lines 441-444: __getattr__ special method routing."""
        from openbrowser.config import Config

        config = Config()
        # Test get_default_profile
        fn = config.get_default_profile
        assert callable(fn)

        # Test get_default_llm
        fn = config.get_default_llm
        assert callable(fn)

        # Test get_default_agent
        fn = config.get_default_agent
        assert callable(fn)

        # Test load_config
        fn = config.load_config
        assert callable(fn)

        # Note: _ensure_dirs starts with '_' so __getattr__ blocks it at line 418.
        # That branch (line 441) is unreachable dead code.

    def test_getattr_unknown(self):
        """Line 444: __getattr__ raises AttributeError for unknown."""
        from openbrowser.config import Config

        config = Config()
        with pytest.raises(AttributeError):
            _ = config.TOTALLY_UNKNOWN_ATTRIBUTE

    def test_getattr_private(self):
        """Line 418: __getattr__ raises for private attrs."""
        from openbrowser.config import Config

        config = Config()
        with pytest.raises(AttributeError):
            _ = config._unknown_private

    def test_get_config_path_with_config_path(self):
        """Line 450: _get_config_path with OPENBROWSER_CONFIG_PATH."""
        from openbrowser.config import Config

        config = Config()
        with patch.object(
            Config,
            "_get_env_config",
            return_value=MagicMock(
                OPENBROWSER_CONFIG_PATH="/tmp/test.json",
                OPENBROWSER_CONFIG_DIR=None,
                XDG_CONFIG_HOME="~/.config",
            ),
        ):
            path = config._get_config_path()
            assert "test.json" in str(path)

    def test_get_config_path_with_config_dir(self):
        """Line 452: _get_config_path with OPENBROWSER_CONFIG_DIR."""
        from openbrowser.config import Config

        config = Config()
        with patch.object(
            Config,
            "_get_env_config",
            return_value=MagicMock(
                OPENBROWSER_CONFIG_PATH=None,
                OPENBROWSER_CONFIG_DIR="/tmp/test_dir",
                XDG_CONFIG_HOME="~/.config",
            ),
        ):
            path = config._get_config_path()
            assert "test_dir" in str(path)
            assert str(path).endswith("config.json")

    def test_get_config_path_default(self):
        """Lines 453-455: _get_config_path with default XDG."""
        from openbrowser.config import Config

        config = Config()
        with patch.object(
            Config,
            "_get_env_config",
            return_value=MagicMock(
                OPENBROWSER_CONFIG_PATH=None,
                OPENBROWSER_CONFIG_DIR=None,
                XDG_CONFIG_HOME="~/.config",
            ),
        ):
            path = config._get_config_path()
            assert "openbrowser" in str(path)
            assert str(path).endswith("config.json")

    def test_get_default_profile_no_default(self):
        """Lines 470-473: _get_default_profile with no default profile."""
        from openbrowser.config import Config, DBStyleConfigJSON, BrowserProfileEntry

        config = Config()
        profile_id = "test-id"
        db_config = DBStyleConfigJSON()
        db_config.browser_profile[profile_id] = BrowserProfileEntry(
            id=profile_id, default=False
        )
        with patch.object(Config, "_get_db_config", return_value=db_config):
            result = config._get_default_profile()
            assert "id" in result

    def test_get_default_profile_empty(self):
        """Lines 472-473: _get_default_profile with empty profiles."""
        from openbrowser.config import Config, DBStyleConfigJSON

        config = Config()
        db_config = DBStyleConfigJSON()
        with patch.object(Config, "_get_db_config", return_value=db_config):
            result = config._get_default_profile()
            assert result == {}

    def test_get_default_llm_no_default(self):
        """Lines 483-486: _get_default_llm with no default."""
        from openbrowser.config import Config, DBStyleConfigJSON, LLMEntry

        config = Config()
        llm_id = "test-llm"
        db_config = DBStyleConfigJSON()
        db_config.llm[llm_id] = LLMEntry(id=llm_id, default=False)
        with patch.object(Config, "_get_db_config", return_value=db_config):
            result = config._get_default_llm()
            assert "id" in result

    def test_get_default_llm_empty(self):
        """Lines 485-486: _get_default_llm with empty."""
        from openbrowser.config import Config, DBStyleConfigJSON

        config = Config()
        db_config = DBStyleConfigJSON()
        with patch.object(Config, "_get_db_config", return_value=db_config):
            result = config._get_default_llm()
            assert result == {}

    def test_get_default_agent_no_default(self):
        """Lines 496-499: _get_default_agent with no default."""
        from openbrowser.config import Config, DBStyleConfigJSON, AgentEntry

        config = Config()
        agent_id = "test-agent"
        db_config = DBStyleConfigJSON()
        db_config.agent[agent_id] = AgentEntry(id=agent_id, default=False)
        with patch.object(Config, "_get_db_config", return_value=db_config):
            result = config._get_default_agent()
            assert "id" in result

    def test_get_default_agent_empty(self):
        """Lines 498-499: _get_default_agent with empty."""
        from openbrowser.config import Config, DBStyleConfigJSON

        config = Config()
        db_config = DBStyleConfigJSON()
        with patch.object(Config, "_get_db_config", return_value=db_config):
            result = config._get_default_agent()
            assert result == {}

    def test_load_config_with_overrides(self):
        """Lines 514-540: _load_config with env var overrides."""
        from openbrowser.config import Config

        config = Config()
        mock_env = MagicMock()
        mock_env.OPENBROWSER_HEADLESS = True
        mock_env.OPENBROWSER_ALLOWED_DOMAINS = "example.com, test.com"
        mock_env.OPENBROWSER_LLM_MODEL = "gpt-4"
        mock_env.OPENAI_API_KEY = "test-key"
        mock_env.OPENBROWSER_PROXY_URL = "http://proxy:8080"
        mock_env.OPENBROWSER_NO_PROXY = "localhost, 127.0.0.1"
        mock_env.OPENBROWSER_PROXY_USERNAME = "user"
        mock_env.OPENBROWSER_PROXY_PASSWORD = "pass"

        with patch.object(Config, "_get_env_config", return_value=mock_env):
            with patch.object(
                Config,
                "_get_default_profile",
                return_value={"headless": False},
            ):
                with patch.object(
                    Config,
                    "_get_default_llm",
                    return_value={"model": "gpt-3.5"},
                ):
                    with patch.object(
                        Config,
                        "_get_default_agent",
                        return_value={"max_steps": 10},
                    ):
                        result = config._load_config()
                        assert result["browser_profile"]["headless"] is True
                        assert "example.com" in result["browser_profile"]["allowed_domains"]
                        assert result["llm"]["model"] == "gpt-4"
                        assert result["llm"]["api_key"] == "test-key"
                        assert result["browser_profile"]["proxy"]["server"] == "http://proxy:8080"
                        assert result["browser_profile"]["proxy"]["username"] == "user"
                        assert result["browser_profile"]["proxy"]["password"] == "pass"


class TestHelperFunctions:
    """Test module-level helper functions."""

    def test_load_openbrowser_config(self, tmp_path):
        """Test load_openbrowser_config with isolated config path."""
        from openbrowser.config import load_openbrowser_config

        # Redirect config to tmp_path so we never touch the real user config
        with patch.dict(os.environ, {
            "OPENBROWSER_CONFIG_DIR": str(tmp_path / "config"),
            "XDG_CONFIG_HOME": str(tmp_path / "xdg"),
        }):
            result = load_openbrowser_config()
        assert isinstance(result, dict)
        assert "browser_profile" in result

    def test_get_default_profile_helper(self):
        """Test get_default_profile helper."""
        from openbrowser.config import get_default_profile

        config = {"browser_profile": {"headless": True}, "llm": {}}
        result = get_default_profile(config)
        assert result["headless"] is True

    def test_get_default_llm_helper(self):
        """Test get_default_llm helper."""
        from openbrowser.config import get_default_llm

        config = {"llm": {"model": "gpt-4"}, "browser_profile": {}}
        result = get_default_llm(config)
        assert result["model"] == "gpt-4"

    def test_get_default_profile_empty(self):
        """Test get_default_profile with empty config."""
        from openbrowser.config import get_default_profile

        result = get_default_profile({})
        assert result == {}

    def test_get_default_llm_empty(self):
        """Test get_default_llm with empty config."""
        from openbrowser.config import get_default_llm

        result = get_default_llm({})
        assert result == {}


class TestDBStyleModels:
    """Test Pydantic models."""

    def test_db_style_entry_defaults(self):
        """Test DBStyleEntry defaults."""
        from openbrowser.config import DBStyleEntry

        entry = DBStyleEntry()
        assert entry.id is not None
        assert entry.default is False
        assert entry.created_at is not None

    def test_browser_profile_entry(self):
        """Test BrowserProfileEntry."""
        from openbrowser.config import BrowserProfileEntry

        entry = BrowserProfileEntry(headless=True, user_data_dir="/tmp")
        assert entry.headless is True
        assert entry.user_data_dir == "/tmp"

    def test_llm_entry(self):
        """Test LLMEntry."""
        from openbrowser.config import LLMEntry

        entry = LLMEntry(model="gpt-4", api_key="test")
        assert entry.model == "gpt-4"

    def test_agent_entry(self):
        """Test AgentEntry."""
        from openbrowser.config import AgentEntry

        entry = AgentEntry(max_steps=10, use_vision=True)
        assert entry.max_steps == 10

    def test_create_default_config(self):
        """Test create_default_config."""
        from openbrowser.config import create_default_config

        config = create_default_config()
        assert len(config.browser_profile) == 1
        assert len(config.llm) == 1
        assert len(config.agent) == 1

    def test_flat_env_config(self):
        """Test FlatEnvConfig."""
        from openbrowser.config import FlatEnvConfig

        # Use controlled env to avoid interference from real env vars
        env_override = {
            "OPENBROWSER_LOGGING_LEVEL": "info",
            "CDP_LOGGING_LEVEL": "WARNING",
            "ANONYMIZED_TELEMETRY": "true",
        }
        with patch.dict(os.environ, env_override, clear=False):
            config = FlatEnvConfig()
            assert config.OPENBROWSER_LOGGING_LEVEL == "info"
            assert config.CDP_LOGGING_LEVEL == "WARNING"
            assert config.ANONYMIZED_TELEMETRY is True
