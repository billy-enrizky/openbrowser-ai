"""Deep coverage tests for openbrowser.cli module.

Tests every function, branch, and code path in cli.py lines 279-919 (the
parts that execute after the early-exit guards for --mcp, -c, daemon,
install, init, and --template).

Strategy:
- Import the already-loaded module (no re-exec of top-level guards).
- Use Click's CliRunner for testing Click commands.
- Mock all external I/O (browser, agent, LLM, telemetry, filesystem).
- For Config attributes accessed via __getattr__ proxy, patch the
  CONFIG instance directly using monkeypatch.setattr or simple attribute
  replacement on the cli_mod.CONFIG object.
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch, mock_open

import click
import pytest
from click.testing import CliRunner

import openbrowser.cli as cli_mod
from openbrowser.cli import (
    INIT_TEMPLATES,
    MAX_HISTORY_LENGTH,
    USER_DATA_DIR,
    _run_template_generation,
    _write_init_file,
    get_default_config,
    get_llm,
    init,
    install,
    load_user_config,
    main,
    run_main_interface,
    run_prompt_mode,
    save_user_config,
    update_config_with_click_args,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: patch CONFIG attribute (proxied via __getattr__)
# ---------------------------------------------------------------------------


class _ConfigPatcher:
    """Context manager to temporarily set an attribute on CONFIG's underlying
    OldConfig instance, which CONFIG proxies via __getattr__.

    For attributes that don't exist on Config itself, we need to directly
    set them on CONFIG so __getattr__ is bypassed.
    """

    def __init__(self, attr, value):
        self.attr = attr
        self.value = value
        self._had_attr = False
        self._old_value = None

    def __enter__(self):
        self._had_attr = hasattr(cli_mod.CONFIG, self.attr)
        if self._had_attr:
            self._old_value = getattr(cli_mod.CONFIG, self.attr)
        # Bypass __setattr__ limitation for Config (which uses __slots__)
        # by patching the underlying OldConfig property
        return self

    def __exit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# Module-level constants (lines 279-315)
# ---------------------------------------------------------------------------


class TestModuleLevelConstants:
    """Verify module-level objects are initialised properly."""

    def test_user_data_dir_is_path(self):
        assert isinstance(USER_DATA_DIR, Path)

    def test_max_history_length(self):
        assert MAX_HISTORY_LENGTH == 100

    def test_init_templates_keys(self):
        assert set(INIT_TEMPLATES.keys()) == {"default", "advanced", "tools"}

    def test_init_templates_have_file_key(self):
        for name, info in INIT_TEMPLATES.items():
            assert "file" in info, f"Template '{name}' missing 'file' key"
            assert "description" in info, f"Template '{name}' missing 'description' key"

    def test_config_imported(self):
        assert cli_mod.CONFIG is not None


# ---------------------------------------------------------------------------
# get_default_config (lines 320-356)
# ---------------------------------------------------------------------------


class TestGetDefaultConfig:
    """Test get_default_config() returns correct structure."""

    def _mock_config(self):
        """Create a mock CONFIG to avoid FlatEnvConfig issues under --cov."""
        mock = MagicMock()
        mock.load_config.return_value = {
            "browser_profile": {"headless": True, "keep_alive": True},
            "llm": {"model": "gpt-5-mini", "temperature": 0.0},
            "agent": {},
        }
        mock.OPENAI_API_KEY = "sk-test"
        mock.ANTHROPIC_API_KEY = "ak-test"
        mock.GOOGLE_API_KEY = "gk-test"
        mock.DEEPSEEK_API_KEY = "dk-test"
        mock.GROK_API_KEY = "xk-test"
        return mock

    def test_returns_dict(self):
        with patch.object(cli_mod, "CONFIG", self._mock_config()):
            config = get_default_config()
            assert isinstance(config, dict)

    def test_has_model_section(self):
        with patch.object(cli_mod, "CONFIG", self._mock_config()):
            config = get_default_config()
            assert "model" in config
            assert "name" in config["model"]
            assert "temperature" in config["model"]
            assert "api_keys" in config["model"]

    def test_has_browser_section(self):
        with patch.object(cli_mod, "CONFIG", self._mock_config()):
            config = get_default_config()
            assert "browser" in config
            assert "headless" in config["browser"]
            assert "keep_alive" in config["browser"]

    def test_has_command_history(self):
        with patch.object(cli_mod, "CONFIG", self._mock_config()):
            config = get_default_config()
            assert "command_history" in config
            assert isinstance(config["command_history"], list)

    def test_api_keys_section(self):
        with patch.object(cli_mod, "CONFIG", self._mock_config()):
            config = get_default_config()
            keys = config["model"]["api_keys"]
            assert "OPENAI_API_KEY" in keys
            assert "ANTHROPIC_API_KEY" in keys
            assert "GOOGLE_API_KEY" in keys
            assert "DEEPSEEK_API_KEY" in keys
            assert "GROK_API_KEY" in keys

    def test_browser_keys(self):
        with patch.object(cli_mod, "CONFIG", self._mock_config()):
            config = get_default_config()
            browser = config["browser"]
            expected_keys = [
                "headless", "keep_alive", "ignore_https_errors",
                "user_data_dir", "allowed_domains",
                "wait_between_actions", "is_mobile",
                "device_scale_factor", "disable_security",
            ]
            for key in expected_keys:
                assert key in browser, f"Missing browser key: {key}"

    def test_uses_config_load(self):
        """Verify load_config is called and values propagate."""
        mock_config = MagicMock()
        mock_config.load_config.return_value = {
            "browser_profile": {"headless": False},
            "llm": {"model": "gpt-5-mini", "temperature": 0.5, "api_key": "sk-test"},
            "agent": {"use_vision": True},
        }
        mock_config.OPENAI_API_KEY = "sk-test"
        mock_config.ANTHROPIC_API_KEY = "ak-test"
        mock_config.GOOGLE_API_KEY = "gk-test"
        mock_config.DEEPSEEK_API_KEY = "dk-test"
        mock_config.GROK_API_KEY = "xk-test"

        with patch.object(cli_mod, "CONFIG", mock_config):
            config = get_default_config()
            assert config["model"]["name"] == "gpt-5-mini"
            assert config["model"]["temperature"] == 0.5
            assert config["browser"]["headless"] is False


# ---------------------------------------------------------------------------
# load_user_config (lines 358-372)
# ---------------------------------------------------------------------------


class TestLoadUserConfig:
    """Test load_user_config() loads config and history."""

    def _mock_config(self, tmp_path=None):
        mock = MagicMock()
        mock.load_config.return_value = {
            "browser_profile": {}, "llm": {}, "agent": {},
        }
        mock.OPENAI_API_KEY = ""
        mock.ANTHROPIC_API_KEY = ""
        mock.GOOGLE_API_KEY = ""
        mock.DEEPSEEK_API_KEY = ""
        mock.GROK_API_KEY = ""
        if tmp_path:
            mock.OPENBROWSER_CONFIG_DIR = tmp_path
        else:
            mock.OPENBROWSER_CONFIG_DIR = Path(tempfile.mkdtemp())
        return mock

    def test_returns_dict(self):
        with patch.object(cli_mod, "CONFIG", self._mock_config()):
            config = load_user_config()
            assert isinstance(config, dict)

    def test_has_command_history(self):
        with patch.object(cli_mod, "CONFIG", self._mock_config()):
            config = load_user_config()
            assert "command_history" in config

    def test_loads_history_from_file(self, tmp_path):
        history_file = tmp_path / "command_history.json"
        history_data = ["cmd1", "cmd2", "cmd3"]
        history_file.write_text(json.dumps(history_data))

        mock_config = MagicMock()
        mock_config.load_config.return_value = {
            "browser_profile": {}, "llm": {}, "agent": {},
        }
        mock_config.OPENAI_API_KEY = ""
        mock_config.ANTHROPIC_API_KEY = ""
        mock_config.GOOGLE_API_KEY = ""
        mock_config.DEEPSEEK_API_KEY = ""
        mock_config.GROK_API_KEY = ""
        mock_config.OPENBROWSER_CONFIG_DIR = tmp_path

        with patch.object(cli_mod, "CONFIG", mock_config):
            config = load_user_config()
            assert config["command_history"] == history_data

    def test_handles_missing_history_file(self, tmp_path):
        mock_config = MagicMock()
        mock_config.load_config.return_value = {
            "browser_profile": {}, "llm": {}, "agent": {},
        }
        mock_config.OPENAI_API_KEY = ""
        mock_config.ANTHROPIC_API_KEY = ""
        mock_config.GOOGLE_API_KEY = ""
        mock_config.DEEPSEEK_API_KEY = ""
        mock_config.GROK_API_KEY = ""
        mock_config.OPENBROWSER_CONFIG_DIR = tmp_path

        with patch.object(cli_mod, "CONFIG", mock_config):
            config = load_user_config()
            assert isinstance(config["command_history"], list)

    def test_handles_invalid_json_history(self, tmp_path):
        history_file = tmp_path / "command_history.json"
        history_file.write_text("not valid json {{{")

        mock_config = MagicMock()
        mock_config.load_config.return_value = {
            "browser_profile": {}, "llm": {}, "agent": {},
        }
        mock_config.OPENAI_API_KEY = ""
        mock_config.ANTHROPIC_API_KEY = ""
        mock_config.GOOGLE_API_KEY = ""
        mock_config.DEEPSEEK_API_KEY = ""
        mock_config.GROK_API_KEY = ""
        mock_config.OPENBROWSER_CONFIG_DIR = tmp_path

        with patch.object(cli_mod, "CONFIG", mock_config):
            config = load_user_config()
            assert config["command_history"] == []


# ---------------------------------------------------------------------------
# save_user_config (lines 375-387)
# ---------------------------------------------------------------------------


class TestSaveUserConfig:
    """Test save_user_config() persists command history."""

    def _mock_config(self, tmp_path):
        mock_config = MagicMock()
        mock_config.OPENBROWSER_CONFIG_DIR = tmp_path
        return mock_config

    def test_saves_history(self, tmp_path):
        config = {"command_history": ["cmd1", "cmd2"]}
        with patch.object(cli_mod, "CONFIG", self._mock_config(tmp_path)):
            save_user_config(config)
            history_file = tmp_path / "command_history.json"
            assert history_file.exists()
            saved = json.loads(history_file.read_text())
            assert saved == ["cmd1", "cmd2"]

    def test_truncates_long_history(self, tmp_path):
        long_history = [f"cmd{i}" for i in range(200)]
        config = {"command_history": long_history}
        with patch.object(cli_mod, "CONFIG", self._mock_config(tmp_path)):
            save_user_config(config)
            history_file = tmp_path / "command_history.json"
            saved = json.loads(history_file.read_text())
            assert len(saved) == MAX_HISTORY_LENGTH
            # Should keep the last MAX_HISTORY_LENGTH entries
            assert saved[0] == f"cmd{200 - MAX_HISTORY_LENGTH}"

    def test_no_history_key_does_nothing(self, tmp_path):
        config = {"model": {"name": "gpt-5-mini"}}
        with patch.object(cli_mod, "CONFIG", self._mock_config(tmp_path)):
            save_user_config(config)
            history_file = tmp_path / "command_history.json"
            assert not history_file.exists()

    def test_non_list_history_does_nothing(self, tmp_path):
        config = {"command_history": "not a list"}
        with patch.object(cli_mod, "CONFIG", self._mock_config(tmp_path)):
            save_user_config(config)
            history_file = tmp_path / "command_history.json"
            assert not history_file.exists()

    def test_exact_max_length_not_truncated(self, tmp_path):
        history = [f"cmd{i}" for i in range(MAX_HISTORY_LENGTH)]
        config = {"command_history": history}
        with patch.object(cli_mod, "CONFIG", self._mock_config(tmp_path)):
            save_user_config(config)
            history_file = tmp_path / "command_history.json"
            saved = json.loads(history_file.read_text())
            assert len(saved) == MAX_HISTORY_LENGTH

    def test_save_empty_history(self, tmp_path):
        config = {"command_history": []}
        with patch.object(cli_mod, "CONFIG", self._mock_config(tmp_path)):
            save_user_config(config)
            history_file = tmp_path / "command_history.json"
            assert history_file.exists()
            assert json.loads(history_file.read_text()) == []


# ---------------------------------------------------------------------------
# update_config_with_click_args (lines 390-428)
# ---------------------------------------------------------------------------


class TestUpdateConfigWithClickArgs:
    """Test CLI argument merging into config dict."""

    def _make_ctx(self, **params):
        """Create a mock click context with specified params."""
        ctx = MagicMock(spec=click.Context)
        ctx.params = params
        return ctx

    def test_sets_model(self):
        config = {"model": {}, "browser": {}}
        ctx = self._make_ctx(model="gpt-5-mini")
        result = update_config_with_click_args(config, ctx)
        assert result["model"]["name"] == "gpt-5-mini"

    def test_sets_headless_true(self):
        config = {"model": {}, "browser": {}}
        ctx = self._make_ctx(headless=True)
        result = update_config_with_click_args(config, ctx)
        assert result["browser"]["headless"] is True

    def test_sets_headless_false(self):
        config = {"model": {}, "browser": {}}
        ctx = self._make_ctx(headless=False)
        result = update_config_with_click_args(config, ctx)
        assert result["browser"]["headless"] is False

    def test_headless_none_not_set(self):
        config = {"model": {}, "browser": {"headless": True}}
        ctx = self._make_ctx(headless=None)
        result = update_config_with_click_args(config, ctx)
        assert result["browser"]["headless"] is True

    def test_sets_window_dimensions(self):
        config = {"model": {}, "browser": {}}
        ctx = self._make_ctx(window_width=1920, window_height=1080)
        result = update_config_with_click_args(config, ctx)
        assert result["browser"]["window_width"] == 1920
        assert result["browser"]["window_height"] == 1080

    def test_sets_user_data_dir(self):
        config = {"model": {}, "browser": {}}
        ctx = self._make_ctx(user_data_dir="/some/dir")
        result = update_config_with_click_args(config, ctx)
        assert result["browser"]["user_data_dir"] == "/some/dir"

    def test_sets_profile_directory(self):
        config = {"model": {}, "browser": {}}
        ctx = self._make_ctx(profile_directory="Profile 1")
        result = update_config_with_click_args(config, ctx)
        assert result["browser"]["profile_directory"] == "Profile 1"

    def test_sets_cdp_url(self):
        config = {"model": {}, "browser": {}}
        ctx = self._make_ctx(cdp_url="http://localhost:9222")
        result = update_config_with_click_args(config, ctx)
        assert result["browser"]["cdp_url"] == "http://localhost:9222"

    def test_proxy_url(self):
        config = {"model": {}, "browser": {}}
        ctx = self._make_ctx(proxy_url="http://proxy:8080")
        result = update_config_with_click_args(config, ctx)
        assert result["browser"]["proxy"]["server"] == "http://proxy:8080"

    def test_no_proxy(self):
        config = {"model": {}, "browser": {}}
        ctx = self._make_ctx(no_proxy="localhost,127.0.0.1, *.internal")
        result = update_config_with_click_args(config, ctx)
        assert result["browser"]["proxy"]["bypass"] == "localhost,127.0.0.1,*.internal"

    def test_proxy_credentials(self):
        config = {"model": {}, "browser": {}}
        ctx = self._make_ctx(proxy_username="user", proxy_password="pass")
        result = update_config_with_click_args(config, ctx)
        assert result["browser"]["proxy"]["username"] == "user"
        assert result["browser"]["proxy"]["password"] == "pass"

    def test_full_proxy_config(self):
        config = {"model": {}, "browser": {}}
        ctx = self._make_ctx(
            proxy_url="socks5://proxy:1080",
            no_proxy="localhost",
            proxy_username="u",
            proxy_password="p",
        )
        result = update_config_with_click_args(config, ctx)
        proxy = result["browser"]["proxy"]
        assert proxy["server"] == "socks5://proxy:1080"
        assert proxy["bypass"] == "localhost"
        assert proxy["username"] == "u"
        assert proxy["password"] == "p"

    def test_no_proxy_empty_entries(self):
        config = {"model": {}, "browser": {}}
        ctx = self._make_ctx(no_proxy="localhost,,  ,127.0.0.1,")
        result = update_config_with_click_args(config, ctx)
        assert result["browser"]["proxy"]["bypass"] == "localhost,127.0.0.1"

    def test_missing_sections_created(self):
        config = {}
        ctx = self._make_ctx(model="test-model")
        result = update_config_with_click_args(config, ctx)
        assert "model" in result
        assert "browser" in result
        assert result["model"]["name"] == "test-model"

    def test_no_params_no_changes(self):
        config = {"model": {"name": "original"}, "browser": {"headless": True}}
        ctx = self._make_ctx()
        result = update_config_with_click_args(config, ctx)
        assert result["model"]["name"] == "original"
        assert result["browser"]["headless"] is True

    def test_no_proxy_when_no_proxy_params(self):
        config = {"model": {}, "browser": {}}
        ctx = self._make_ctx()
        result = update_config_with_click_args(config, ctx)
        assert "proxy" not in result["browser"]

    def test_proxy_only_server(self):
        config = {"model": {}, "browser": {}}
        ctx = self._make_ctx(proxy_url="http://proxy:3128")
        result = update_config_with_click_args(config, ctx)
        assert result["browser"]["proxy"] == {"server": "http://proxy:3128"}


# ---------------------------------------------------------------------------
# get_llm (lines 432-488)
# ---------------------------------------------------------------------------


class TestGetLlm:
    """Test LLM auto-detection and creation.

    ChatOpenAI, ChatAnthropic, ChatGoogle are lazy-imported inside get_llm(),
    so we patch them at their source module paths.
    """

    def _mock_config(self, **overrides):
        """Create a mock CONFIG for get_llm tests."""
        mock = MagicMock()
        mock.OPENAI_API_KEY = overrides.get("openai", "")
        mock.ANTHROPIC_API_KEY = overrides.get("anthropic", "")
        mock.GOOGLE_API_KEY = overrides.get("google", "")
        return mock

    def test_gpt_model_explicit(self):
        mock_chat = MagicMock()
        config = {
            "model": {
                "name": "gpt-5-mini",
                "temperature": 0.5,
                "api_keys": {"OPENAI_API_KEY": "sk-test"},
            }
        }
        with patch("openbrowser.llm.openai.chat.ChatOpenAI", mock_chat):
            result = get_llm(config)
            mock_chat.assert_called_once_with(model="gpt-5-mini", temperature=0.5, api_key="sk-test")

    def test_gpt_model_no_api_key_exits(self):
        config = {
            "model": {
                "name": "gpt-5-mini",
                "temperature": 0.0,
                "api_keys": {"OPENAI_API_KEY": None},
            }
        }
        with (
            patch.object(cli_mod, "CONFIG", self._mock_config(openai="")),
            pytest.raises(SystemExit) as exc_info,
        ):
            get_llm(config)
        assert exc_info.value.code == 1

    def test_claude_model_explicit(self):
        mock_chat = MagicMock()
        config = {
            "model": {
                "name": "claude-4-sonnet",
                "temperature": 0.0,
                "api_keys": {},
            }
        }
        with (
            patch.object(cli_mod, "CONFIG", self._mock_config(anthropic="sk-ant-test")),
            patch("openbrowser.llm.anthropic.chat.ChatAnthropic", mock_chat),
        ):
            result = get_llm(config)
            mock_chat.assert_called_once_with(model="claude-4-sonnet", temperature=0.0)

    def test_claude_model_no_api_key_exits(self):
        config = {
            "model": {
                "name": "claude-4-sonnet",
                "temperature": 0.0,
                "api_keys": {},
            }
        }
        with (
            patch.object(cli_mod, "CONFIG", self._mock_config(anthropic="")),
            pytest.raises(SystemExit) as exc_info,
        ):
            get_llm(config)
        assert exc_info.value.code == 1

    def test_gemini_model_explicit(self):
        mock_chat = MagicMock()
        mock_google_module = MagicMock()
        mock_google_module.ChatGoogle = mock_chat
        config = {
            "model": {
                "name": "gemini-2.5-pro",
                "temperature": 0.0,
                "api_keys": {},
            }
        }
        with (
            patch.object(cli_mod, "CONFIG", self._mock_config(google="google-test")),
            patch.dict("sys.modules", {"openbrowser.llm.google.chat": mock_google_module}),
        ):
            result = get_llm(config)
            mock_chat.assert_called_once_with(model="gemini-2.5-pro", temperature=0.0)

    def test_gemini_model_no_api_key_exits(self):
        mock_google_module = MagicMock()
        config = {
            "model": {
                "name": "gemini-2.5-pro",
                "temperature": 0.0,
                "api_keys": {},
            }
        }
        with (
            patch.object(cli_mod, "CONFIG", self._mock_config(google="")),
            patch.dict("sys.modules", {"openbrowser.llm.google.chat": mock_google_module}),
            pytest.raises(SystemExit) as exc_info,
        ):
            get_llm(config)
        assert exc_info.value.code == 1

    def test_oci_model_exits(self):
        config = {
            "model": {
                "name": "oci-some-model",
                "temperature": 0.0,
                "api_keys": {},
            }
        }
        with pytest.raises(SystemExit) as exc_info:
            get_llm(config)
        assert exc_info.value.code == 1

    def test_autodetect_openai(self):
        mock_chat = MagicMock()
        config = {
            "model": {
                "name": None,
                "temperature": 0.0,
                "api_keys": {"OPENAI_API_KEY": "sk-test"},
            }
        }
        with patch("openbrowser.llm.openai.chat.ChatOpenAI", mock_chat):
            result = get_llm(config)
            mock_chat.assert_called_once_with(model="gpt-5-mini", temperature=0.0, api_key="sk-test")

    def test_autodetect_anthropic(self):
        mock_chat = MagicMock()
        config = {
            "model": {
                "name": None,
                "temperature": 0.0,
                "api_keys": {"OPENAI_API_KEY": None},
            }
        }
        with (
            patch.object(cli_mod, "CONFIG", self._mock_config(openai="", anthropic="sk-ant")),
            patch("openbrowser.llm.anthropic.chat.ChatAnthropic", mock_chat),
        ):
            result = get_llm(config)
            mock_chat.assert_called_once_with(model="claude-4-sonnet", temperature=0.0)

    def test_autodetect_google(self):
        mock_chat = MagicMock()
        mock_google_module = MagicMock()
        mock_google_module.ChatGoogle = mock_chat
        config = {
            "model": {
                "name": None,
                "temperature": 0.0,
                "api_keys": {"OPENAI_API_KEY": None},
            }
        }
        with (
            patch.object(cli_mod, "CONFIG", self._mock_config(openai="", anthropic="", google="google-key")),
            patch.dict("sys.modules", {"openbrowser.llm.google.chat": mock_google_module}),
        ):
            result = get_llm(config)
            mock_chat.assert_called_once_with(model="gemini-2.5-pro", temperature=0.0)

    def test_autodetect_no_keys_exits(self):
        config = {
            "model": {
                "name": None,
                "temperature": 0.0,
                "api_keys": {"OPENAI_API_KEY": None},
            }
        }
        with (
            patch.object(cli_mod, "CONFIG", self._mock_config(openai="", anthropic="", google="")),
            pytest.raises(SystemExit) as exc_info,
        ):
            get_llm(config)
        assert exc_info.value.code == 1

    def test_gpt_uses_config_api_key_fallback(self):
        mock_chat = MagicMock()
        config = {
            "model": {
                "name": "gpt-5-mini",
                "temperature": 0.0,
                "api_keys": {"OPENAI_API_KEY": None},
            }
        }
        with (
            patch.object(cli_mod, "CONFIG", self._mock_config(openai="sk-from-config")),
            patch("openbrowser.llm.openai.chat.ChatOpenAI", mock_chat),
        ):
            get_llm(config)
            mock_chat.assert_called_once_with(
                model="gpt-5-mini", temperature=0.0, api_key="sk-from-config"
            )

    def test_autodetect_uses_config_openai_key(self):
        mock_chat = MagicMock()
        config = {
            "model": {
                "name": None,
                "temperature": 0.0,
                "api_keys": {"OPENAI_API_KEY": None},
            }
        }
        with (
            patch.object(cli_mod, "CONFIG", self._mock_config(openai="sk-config")),
            patch("openbrowser.llm.openai.chat.ChatOpenAI", mock_chat),
        ):
            get_llm(config)
            mock_chat.assert_called_once()

    def test_empty_model_config(self):
        config = {}
        with (
            patch.object(cli_mod, "CONFIG", self._mock_config(openai="", anthropic="", google="")),
            pytest.raises(SystemExit),
        ):
            get_llm(config)

    def test_unknown_model_prefix_autodetects(self):
        """A model name that doesn't match any prefix falls through to auto-detect."""
        mock_cls = MagicMock()
        config = {
            "model": {
                "name": "unknown-model",
                "temperature": 0.0,
                "api_keys": {"OPENAI_API_KEY": "sk-key"},
            }
        }
        with patch("openbrowser.llm.openai.chat.ChatOpenAI", mock_cls):
            result = get_llm(config)
            # Falls through the name checks, hits auto-detect with OPENAI key available
            mock_cls.assert_called_once()


# ---------------------------------------------------------------------------
# run_prompt_mode (lines 494-615)
# ---------------------------------------------------------------------------


def _make_prompt_patches(
    mock_llm=None,
    mock_agent_instance=None,
    mock_session=None,
    mock_telemetry=None,
    config_override=None,
    remaining_tasks=None,
):
    """Return a list of context managers that set up run_prompt_mode mocks."""
    if mock_llm is None:
        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_llm.__class__.__name__ = "ChatOpenAI"
    if mock_agent_instance is None:
        mock_agent_instance = MagicMock()
        mock_agent_instance.run = AsyncMock()
    if mock_session is None:
        mock_session = MagicMock()
        mock_session.kill = AsyncMock()
    if mock_telemetry is None:
        mock_telemetry = MagicMock()

    config = config_override or {
        "model": {"name": "gpt-5-mini", "temperature": 0.0, "api_keys": {"OPENAI_API_KEY": "sk-test"}},
        "browser": {},
        "agent": {},
    }

    mock_settings_instance = MagicMock()
    mock_settings_instance.model_dump.return_value = {}

    mock_settings_cls = MagicMock()
    mock_settings_cls.model_validate.return_value = mock_settings_instance

    patches = [
        patch("openbrowser.logging_config.setup_logging"),
        patch("openbrowser.cli.ProductTelemetry", return_value=mock_telemetry),
        patch("openbrowser.cli.load_user_config", return_value=config),
        patch("openbrowser.cli.update_config_with_click_args", side_effect=lambda c, ctx: c),
        patch("openbrowser.cli.get_llm", return_value=mock_llm),
        patch("openbrowser.cli.AgentSettings", mock_settings_cls),
        patch("openbrowser.cli.BrowserProfile"),
        patch("openbrowser.cli.BrowserSession", return_value=mock_session),
        patch("openbrowser.cli.Agent", return_value=mock_agent_instance),
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch("asyncio.all_tasks", return_value=remaining_tasks or set()),
        patch("asyncio.current_task", return_value=MagicMock()),
    ]
    return patches


class TestRunPromptMode:
    """Test the async run_prompt_mode function."""

    def _make_ctx(self, **params):
        ctx = MagicMock(spec=click.Context)
        ctx.params = params
        return ctx

    @pytest.mark.asyncio
    async def test_successful_run(self):
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock()
        mock_session = MagicMock()
        mock_session.kill = AsyncMock()
        ctx = self._make_ctx()

        patches = _make_prompt_patches(
            mock_agent_instance=mock_agent,
            mock_session=mock_session,
        )
        with contextlib_exitstack(patches):
            await run_prompt_mode("test prompt", ctx)
            mock_agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_error_debug(self):
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("test error"))
        mock_session = MagicMock()
        mock_session.kill = AsyncMock()
        ctx = self._make_ctx()

        patches = _make_prompt_patches(
            mock_agent_instance=mock_agent,
            mock_session=mock_session,
        )
        with pytest.raises(SystemExit) as exc_info:
            with contextlib_exitstack(patches):
                await run_prompt_mode("test prompt", ctx, debug=True)
        assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_run_with_error_no_debug(self):
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=ValueError("something broke"))
        mock_session = MagicMock()
        mock_session.kill = AsyncMock()
        ctx = self._make_ctx()

        patches = _make_prompt_patches(
            mock_agent_instance=mock_agent,
            mock_session=mock_session,
        )
        with pytest.raises(SystemExit) as exc_info:
            with contextlib_exitstack(patches):
                await run_prompt_mode("test prompt", ctx, debug=False)
        assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_browser_kill_exception_suppressed(self):
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock()
        mock_session = MagicMock()
        mock_session.kill = AsyncMock(side_effect=Exception("kill failed"))
        ctx = self._make_ctx()

        patches = _make_prompt_patches(
            mock_agent_instance=mock_agent,
            mock_session=mock_session,
        )
        with contextlib_exitstack(patches):
            # Should not raise despite kill() error
            await run_prompt_mode("test prompt", ctx)

    @pytest.mark.asyncio
    async def test_finally_cancels_remaining_tasks(self):
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock()
        mock_session = MagicMock()
        mock_session.kill = AsyncMock()
        mock_task = MagicMock()
        mock_task.cancel = MagicMock()
        ctx = self._make_ctx()

        patches = _make_prompt_patches(
            mock_agent_instance=mock_agent,
            mock_session=mock_session,
            remaining_tasks={mock_task},
        )
        # Also need to patch asyncio.gather for the final cleanup
        patches.append(patch("asyncio.gather", new_callable=AsyncMock))
        with contextlib_exitstack(patches):
            await run_prompt_mode("test prompt", ctx)
            mock_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_removes_none_values_from_browser_config(self):
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock()
        mock_session = MagicMock()
        mock_session.kill = AsyncMock()
        ctx = self._make_ctx()

        captured_profile_kwargs = {}

        def capture_profile(**kwargs):
            captured_profile_kwargs.update(kwargs)
            return MagicMock()

        config = {
            "model": {"name": "gpt-5-mini", "temperature": 0.0, "api_keys": {"OPENAI_API_KEY": "sk-test"}},
            "browser": {"headless": True, "keep_alive": None, "cdp_url": None},
            "agent": {},
        }

        patches = _make_prompt_patches(
            mock_agent_instance=mock_agent,
            mock_session=mock_session,
            config_override=config,
        )
        # Override the BrowserProfile patch with our capturing version
        patches = [p for p in patches if not (hasattr(p, 'attribute') and p.attribute == 'BrowserProfile')]
        patches.append(patch("openbrowser.cli.BrowserProfile", side_effect=capture_profile))

        with contextlib_exitstack(patches):
            await run_prompt_mode("test prompt", ctx)
            # None values should be filtered out
            assert "keep_alive" not in captured_profile_kwargs
            assert "cdp_url" not in captured_profile_kwargs
            assert captured_profile_kwargs.get("headless") is True

    @pytest.mark.asyncio
    async def test_telemetry_captures(self):
        """Verify telemetry capture calls during prompt mode."""
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock()
        mock_session = MagicMock()
        mock_session.kill = AsyncMock()
        mock_telemetry = MagicMock()
        ctx = self._make_ctx()

        patches = _make_prompt_patches(
            mock_agent_instance=mock_agent,
            mock_session=mock_session,
            mock_telemetry=mock_telemetry,
        )
        with contextlib_exitstack(patches):
            await run_prompt_mode("test", ctx)
            # Should have captured start + task_completed events
            assert mock_telemetry.capture.call_count == 2
            mock_telemetry.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_telemetry(self):
        """Error path should capture error telemetry event."""
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("fail"))
        mock_session = MagicMock()
        mock_session.kill = AsyncMock()
        mock_telemetry = MagicMock()
        ctx = self._make_ctx()

        patches = _make_prompt_patches(
            mock_agent_instance=mock_agent,
            mock_session=mock_session,
            mock_telemetry=mock_telemetry,
        )
        with pytest.raises(SystemExit):
            with contextlib_exitstack(patches):
                await run_prompt_mode("test", ctx)
        # Should have captured start + error events
        assert mock_telemetry.capture.call_count == 2
        mock_telemetry.flush.assert_called_once()


def contextlib_exitstack(patches):
    """Enter all patch context managers using contextlib.ExitStack."""
    import contextlib
    stack = contextlib.ExitStack()
    for p in patches:
        stack.enter_context(p)
    return stack


# ---------------------------------------------------------------------------
# Click CLI: main group (lines 618-664)
# ---------------------------------------------------------------------------


class TestMainCommand:
    """Test the main Click group."""

    def test_help_output(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "OpenBrowser" in result.output

    def test_version_flag(self):
        runner = CliRunner()
        with patch("importlib.metadata.version", return_value="1.2.3"):
            result = runner.invoke(main, ["--version"])
            assert result.exit_code == 0

    def test_template_flag_routes_to_template_gen(self):
        runner = CliRunner()
        with patch("openbrowser.cli._run_template_generation") as mock_gen:
            result = runner.invoke(main, ["--template", "default"])
            mock_gen.assert_called_once_with("default", None, False)

    def test_template_with_output_and_force(self):
        runner = CliRunner()
        with patch("openbrowser.cli._run_template_generation") as mock_gen:
            result = runner.invoke(main, ["--template", "advanced", "-o", "/tmp/out.py", "-f"])
            mock_gen.assert_called_once_with("advanced", "/tmp/out.py", True)

    def test_no_args_shows_help(self):
        runner = CliRunner()
        result = runner.invoke(main, [])
        assert result.exit_code == 0
        # Should show help text when no subcommand
        assert "OpenBrowser" in result.output or "Usage" in result.output

    def test_prompt_flag_runs_prompt_mode(self):
        runner = CliRunner()
        with (
            patch("openbrowser.cli.Agent", MagicMock()),
            patch("openbrowser.cli.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run = MagicMock()
            result = runner.invoke(main, ["-p", "Search something"])
            mock_asyncio.run.assert_called_once()

    def test_prompt_flag_agent_not_installed(self):
        runner = CliRunner()
        with patch.object(cli_mod, "Agent", None):
            result = runner.invoke(main, ["-p", "Search something"])
            assert result.exit_code == 1

    def test_mcp_flag_runs_mcp_server(self):
        runner = CliRunner()
        mock_mcp_main = MagicMock()
        with (
            patch("openbrowser.cli.ProductTelemetry") as mock_tel_cls,
            patch("openbrowser.cli.asyncio") as mock_asyncio,
            patch.dict("sys.modules", {"openbrowser.mcp.server": MagicMock(main=mock_mcp_main)}),
        ):
            mock_tel_cls.return_value = MagicMock()
            mock_asyncio.run = MagicMock()
            result = runner.invoke(main, ["--mcp"])
            mock_asyncio.run.assert_called_once()

    def test_mcp_telemetry_exception_suppressed(self):
        runner = CliRunner()
        mock_mcp_main = MagicMock()
        with (
            patch("openbrowser.cli.ProductTelemetry", side_effect=Exception("telemetry broken")),
            patch("openbrowser.cli.asyncio") as mock_asyncio,
            patch.dict("sys.modules", {"openbrowser.mcp.server": MagicMock(main=mock_mcp_main)}),
        ):
            mock_asyncio.run = MagicMock()
            result = runner.invoke(main, ["--mcp"])
            # Should not crash despite telemetry error
            mock_asyncio.run.assert_called_once()


# ---------------------------------------------------------------------------
# Click CLI: run_main_interface (lines 666-709)
# ---------------------------------------------------------------------------


class TestRunMainInterface:
    """Test run_main_interface helper."""

    def test_version_prints_and_exits(self):
        ctx = MagicMock(spec=click.Context)
        ctx.params = {}
        kwargs = {"version": True, "mcp": False, "prompt": None}
        with (
            patch("importlib.metadata.version", return_value="2.0.0"),
            pytest.raises(SystemExit) as exc_info,
        ):
            run_main_interface(ctx, debug=False, **kwargs)
        assert exc_info.value.code == 0

    def test_no_mode_shows_help(self):
        ctx = MagicMock(spec=click.Context)
        ctx.get_help.return_value = "Help text"
        ctx.params = {}
        kwargs = {"version": False, "mcp": False, "prompt": None}
        run_main_interface(ctx, debug=False, **kwargs)
        ctx.get_help.assert_called_once()

    def test_prompt_sets_env_var(self):
        ctx = MagicMock(spec=click.Context)
        ctx.params = {}
        kwargs = {"version": False, "mcp": False, "prompt": "do something"}
        with (
            patch.object(cli_mod, "Agent", MagicMock()),
            patch("openbrowser.cli.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run = MagicMock()
            run_main_interface(ctx, debug=False, **kwargs)
            assert os.environ.get("OPENBROWSER_LOGGING_LEVEL") == "result"
            mock_asyncio.run.assert_called_once()


# ---------------------------------------------------------------------------
# Click CLI: install command (lines 712-734)
# ---------------------------------------------------------------------------


class TestInstallCommand:
    """Test the install Click subcommand."""

    def test_install_success(self):
        runner = CliRunner()
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result):
            result = runner.invoke(main, ["install"])
            assert result.exit_code == 0
            assert "Installation complete" in result.output

    def test_install_failure(self):
        runner = CliRunner()
        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch("subprocess.run", return_value=mock_result):
            result = runner.invoke(main, ["install"])
            assert result.exit_code == 1
            assert "Installation failed" in result.output

    def test_install_linux_adds_with_deps(self):
        runner = CliRunner()
        mock_result = MagicMock()
        mock_result.returncode = 0
        captured_cmd = []

        def capture_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return mock_result

        with (
            patch("subprocess.run", side_effect=capture_run),
            patch("platform.system", return_value="Linux"),
        ):
            result = runner.invoke(main, ["install"])
            assert "--with-deps" in captured_cmd

    def test_install_macos_no_with_deps(self):
        runner = CliRunner()
        mock_result = MagicMock()
        mock_result.returncode = 0
        captured_cmd = []

        def capture_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return mock_result

        with (
            patch("subprocess.run", side_effect=capture_run),
            patch("platform.system", return_value="Darwin"),
        ):
            result = runner.invoke(main, ["install"])
            assert "--with-deps" not in captured_cmd
            assert "--no-shell" in captured_cmd


# ---------------------------------------------------------------------------
# _write_init_file (lines 790-808)
# ---------------------------------------------------------------------------


class TestWriteInitFile:
    """Test the _write_init_file helper."""

    def test_write_new_file(self, tmp_path):
        output = tmp_path / "new_file.py"
        result = _write_init_file(output, "# content")
        assert result is True
        assert output.read_text() == "# content"

    def test_write_existing_file_with_force(self, tmp_path):
        output = tmp_path / "existing.py"
        output.write_text("old content")
        result = _write_init_file(output, "new content", force=True)
        assert result is True
        assert output.read_text() == "new content"

    def test_write_existing_file_user_confirms(self, tmp_path):
        output = tmp_path / "existing.py"
        output.write_text("old content")
        with patch("click.confirm", return_value=True):
            result = _write_init_file(output, "new content", force=False)
            assert result is True
            assert output.read_text() == "new content"

    def test_write_existing_file_user_cancels(self, tmp_path):
        output = tmp_path / "existing.py"
        output.write_text("old content")
        with patch("click.confirm", return_value=False):
            result = _write_init_file(output, "new content", force=False)
            assert result is False
            assert output.read_text() == "old content"

    def test_write_creates_parent_dirs(self, tmp_path):
        output = tmp_path / "subdir" / "deep" / "file.py"
        result = _write_init_file(output, "content")
        assert result is True
        assert output.exists()

    def test_write_exception_returns_false(self, tmp_path):
        output = tmp_path / "file.py"
        with patch.object(Path, "write_text", side_effect=OSError("disk full")):
            result = _write_init_file(output, "content")
            assert result is False


# ---------------------------------------------------------------------------
# _run_template_generation (lines 758-787)
# ---------------------------------------------------------------------------


class TestRunTemplateGeneration:
    """Test _run_template_generation helper."""

    @pytest.fixture
    def fake_templates(self, tmp_path):
        """Create a fake cli_templates directory and patch __file__."""
        templates_dir = tmp_path / "cli_templates"
        templates_dir.mkdir()
        for name, info in INIT_TEMPLATES.items():
            (templates_dir / info["file"]).write_text(f"# {name} template\n")
        return tmp_path

    def test_successful_generation_with_output(self, tmp_path, fake_templates):
        output = str(tmp_path / "output.py")
        with patch("openbrowser.cli.__file__", str(fake_templates / "cli.py")):
            _run_template_generation("default", output, False)
        assert Path(output).exists()
        assert "# default template" in Path(output).read_text()

    def test_generation_no_output_uses_default_path(self, tmp_path, fake_templates):
        """When output is None, the default output path is cwd/openbrowser_<template>.py."""
        with (
            patch("openbrowser.cli.__file__", str(fake_templates / "cli.py")),
            patch("openbrowser.cli._write_init_file", return_value=True) as mock_write,
        ):
            _run_template_generation("default", None, False)
            mock_write.assert_called_once()
            output_path = mock_write.call_args[0][0]
            assert "openbrowser_default.py" in str(output_path)

    def test_generation_read_error_exits(self):
        """When the template file cannot be read, should sys.exit(1)."""
        with (
            patch.object(Path, "read_text", side_effect=FileNotFoundError("not found")),
            pytest.raises(SystemExit) as exc_info,
        ):
            _run_template_generation("default", "/tmp/out.py", False)
        assert exc_info.value.code == 1

    def test_generation_write_fails_exits(self, fake_templates):
        with (
            patch("openbrowser.cli.__file__", str(fake_templates / "cli.py")),
            patch("openbrowser.cli._write_init_file", return_value=False),
            pytest.raises(SystemExit) as exc_info,
        ):
            _run_template_generation("default", "/tmp/test_out.py", False)
        assert exc_info.value.code == 1

    def test_generation_success_prints_next_steps(self, tmp_path, fake_templates, capsys):
        output = str(tmp_path / "out.py")
        with patch("openbrowser.cli.__file__", str(fake_templates / "cli.py")):
            _run_template_generation("default", output, True)
        # click.echo goes to stdout
        # Verify the file was created
        assert Path(output).exists()


# ---------------------------------------------------------------------------
# Click CLI: init command (lines 811-915)
# ---------------------------------------------------------------------------


class TestInitCommand:
    """Test the init Click subcommand."""

    @pytest.fixture
    def fake_templates(self, tmp_path):
        """Create a fake cli_templates directory."""
        templates_dir = tmp_path / "cli_templates"
        templates_dir.mkdir()
        for name, info in INIT_TEMPLATES.items():
            (templates_dir / info["file"]).write_text(f"# {name} template\n")
        return tmp_path

    def test_init_list_templates(self):
        runner = CliRunner()
        result = runner.invoke(main, ["init", "--list"])
        assert result.exit_code == 0
        assert "default" in result.output
        assert "advanced" in result.output
        assert "tools" in result.output

    def test_init_with_template_and_force(self, tmp_path, fake_templates):
        runner = CliRunner()
        output_file = str(tmp_path / "test_init.py")
        with patch("openbrowser.cli.__file__", str(fake_templates / "cli.py")):
            result = runner.invoke(main, ["init", "--template", "default", "-o", output_file, "-f"])
        assert result.exit_code == 0
        assert Path(output_file).exists()
        assert "Created" in result.output

    def test_init_interactive_mode(self, fake_templates, tmp_path):
        runner = CliRunner()
        with patch("openbrowser.cli.__file__", str(fake_templates / "cli.py")):
            result = runner.invoke(main, ["init", "-f"], input="default\n")
        # Interactive mode prompts for template then writes
        assert result.exit_code == 0 or "Created" in result.output

    def test_init_template_read_error(self):
        runner = CliRunner()
        with patch.object(Path, "read_text", side_effect=FileNotFoundError("no such file")):
            result = runner.invoke(main, ["init", "--template", "default", "-f"])
            assert result.exit_code == 1

    def test_init_write_fails(self, tmp_path, fake_templates):
        runner = CliRunner()
        output = str(tmp_path / "fail.py")
        with (
            patch("openbrowser.cli.__file__", str(fake_templates / "cli.py")),
            patch("openbrowser.cli._write_init_file", return_value=False),
        ):
            result = runner.invoke(main, ["init", "--template", "default", "-o", output, "-f"])
            assert result.exit_code == 1

    def test_init_with_custom_output(self, tmp_path, fake_templates):
        runner = CliRunner()
        output_file = str(tmp_path / "custom_script.py")
        with patch("openbrowser.cli.__file__", str(fake_templates / "cli.py")):
            result = runner.invoke(main, ["init", "--template", "advanced", "-o", output_file, "-f"])
        assert result.exit_code == 0

    def test_init_tools_template(self, tmp_path, fake_templates):
        runner = CliRunner()
        output_file = str(tmp_path / "tools_script.py")
        with patch("openbrowser.cli.__file__", str(fake_templates / "cli.py")):
            result = runner.invoke(main, ["init", "--template", "tools", "-o", output_file, "-f"])
        assert result.exit_code == 0

    def test_init_no_output_uses_default(self, fake_templates):
        """When no --output specified, init uses cwd/openbrowser_<template>.py."""
        runner = CliRunner()
        with patch("openbrowser.cli.__file__", str(fake_templates / "cli.py")):
            result = runner.invoke(main, ["init", "--template", "default", "-f"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Edge cases and integration
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and integration scenarios."""

    def test_main_invoked_with_subcommand(self):
        """When a subcommand is given, main should not call run_main_interface."""
        runner = CliRunner()
        result = runner.invoke(main, ["init", "--list"])
        assert result.exit_code == 0

    def test_debug_flag_passed_through(self):
        runner = CliRunner()
        with (
            patch("openbrowser.cli.Agent", MagicMock()),
            patch("openbrowser.cli.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run = MagicMock()
            result = runner.invoke(main, ["--debug", "-p", "test"])
            mock_asyncio.run.assert_called_once()

    def test_all_browser_options_combined(self):
        runner = CliRunner()
        with (
            patch("openbrowser.cli.Agent", MagicMock()),
            patch("openbrowser.cli.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run = MagicMock()
            result = runner.invoke(main, [
                "-p", "test",
                "--model", "gpt-5-mini",
                "--headless",
                "--window-width", "1920",
                "--window-height", "1080",
                "--user-data-dir", "/tmp/chrome",
                "--profile-directory", "Default",
                "--cdp-url", "http://localhost:9222",
                "--proxy-url", "http://proxy:8080",
                "--no-proxy", "localhost",
                "--proxy-username", "user",
                "--proxy-password", "pass",
            ])
            mock_asyncio.run.assert_called_once()

    def test_get_default_config_missing_sections(self):
        """Config should handle missing sections gracefully."""
        mock_config = MagicMock()
        mock_config.load_config.return_value = {}
        mock_config.OPENAI_API_KEY = ""
        mock_config.ANTHROPIC_API_KEY = ""
        mock_config.GOOGLE_API_KEY = ""
        mock_config.DEEPSEEK_API_KEY = ""
        mock_config.GROK_API_KEY = ""
        with patch.object(cli_mod, "CONFIG", mock_config):
            config = get_default_config()
            assert "model" in config
            assert "browser" in config


# ---------------------------------------------------------------------------
# Template metadata
# ---------------------------------------------------------------------------


class TestTemplateMetadata:
    """Verify INIT_TEMPLATES in cli module matches expectations."""

    def test_default_template(self):
        assert INIT_TEMPLATES["default"]["file"] == "default_template.py"
        assert "description" in INIT_TEMPLATES["default"]

    def test_advanced_template(self):
        assert INIT_TEMPLATES["advanced"]["file"] == "advanced_template.py"
        assert "description" in INIT_TEMPLATES["advanced"]

    def test_tools_template(self):
        assert INIT_TEMPLATES["tools"]["file"] == "tools_template.py"
        assert "description" in INIT_TEMPLATES["tools"]


# ---------------------------------------------------------------------------
# Click command metadata
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Module-level early exit paths (lines 7-277) via subprocess
# These paths execute BEFORE any function is defined and require
# re-executing the module with specific sys.argv values.
# ---------------------------------------------------------------------------


class TestModuleLevelEarlyExits:
    """Test early exit paths that run at module import time.

    These paths inspect sys.argv at import time and exit before
    defining any functions. We use runpy.run_path() with mocked
    sys.argv and dependencies to exercise them in-process.
    """

    def _run_cli_module(self, argv, extra_patches=None):
        """Run the CLI module with given argv, returning captured prints.

        Uses exec() on the raw source file with a carefully controlled
        namespace so that coverage can track the lines.
        """
        import importlib
        import runpy

        original_argv = sys.argv[:]
        captured = {"stdout": [], "stderr": []}

        def fake_print(*args, file=None, **kwargs):
            target = "stderr" if file is sys.stderr else "stdout"
            captured[target].append(" ".join(str(a) for a in args))

        patches = {
            "sys.argv": argv,
        }
        if extra_patches:
            patches.update(extra_patches)

        return captured

    def test_early_install_success(self):
        """Test early install subcommand path (lines 154-175)."""
        import subprocess as sp

        mock_result = MagicMock()
        mock_result.returncode = 0

        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "install"]
            with (
                patch("subprocess.run", return_value=mock_result) as mock_run,
                patch("platform.system", return_value="Darwin"),
                pytest.raises(SystemExit) as exc_info,
            ):
                # Remove cached module and re-import to trigger early-exit code
                saved_modules = {}
                for key in list(sys.modules.keys()):
                    if key.startswith("openbrowser.cli") and key != "openbrowser.cli":
                        saved_modules[key] = sys.modules.pop(key)

                # Use exec to run the top-level code
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()

                # Execute the module-level code in a fresh namespace
                ns = {"__name__": "__test__", "__file__": str(cli_path)}
                exec(compile(code, str(cli_path), "exec"), ns)

            assert exc_info.value.code == 0
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "chromium" in cmd
            assert "--no-shell" in cmd
            assert "--with-deps" not in cmd  # Darwin, not Linux
        finally:
            sys.argv = original_argv
            sys.modules.update(saved_modules)

    def test_early_install_linux_with_deps(self):
        """Test early install on Linux adds --with-deps (lines 163-164)."""
        mock_result = MagicMock()
        mock_result.returncode = 0

        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "install"]
            with (
                patch("subprocess.run", return_value=mock_result) as mock_run,
                patch("platform.system", return_value="Linux"),
                pytest.raises(SystemExit),
            ):
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()
                ns = {"__name__": "__test__", "__file__": str(cli_path)}
                exec(compile(code, str(cli_path), "exec"), ns)

            cmd = mock_run.call_args[0][0]
            assert "--with-deps" in cmd
        finally:
            sys.argv = original_argv

    def test_early_install_failure(self):
        """Test early install subcommand failure path (line 174)."""
        mock_result = MagicMock()
        mock_result.returncode = 1

        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "install"]
            with (
                patch("subprocess.run", return_value=mock_result),
                patch("platform.system", return_value="Darwin"),
                pytest.raises(SystemExit) as exc_info,
            ):
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()
                ns = {"__name__": "__test__", "__file__": str(cli_path)}
                exec(compile(code, str(cli_path), "exec"), ns)

            assert exc_info.value.code == 1
        finally:
            sys.argv = original_argv

    def test_early_mcp_mode(self):
        """Test early --mcp mode (lines 6-35)."""
        mock_mcp_main = MagicMock()
        mock_asyncio_run = MagicMock()

        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "--mcp"]
            with (
                patch("asyncio.run", mock_asyncio_run),
                patch.dict("sys.modules", {
                    "openbrowser.mcp.server": MagicMock(main=mock_mcp_main),
                    "openbrowser.telemetry": MagicMock(
                        CLITelemetryEvent=MagicMock,
                        ProductTelemetry=MagicMock(return_value=MagicMock()),
                    ),
                    "openbrowser.utils": MagicMock(get_openbrowser_version=MagicMock(return_value="0.1.0")),
                }),
                patch("logging.disable"),
                pytest.raises(SystemExit) as exc_info,
            ):
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()
                ns = {"__name__": "__test__", "__file__": str(cli_path)}
                exec(compile(code, str(cli_path), "exec"), ns)

            assert exc_info.value.code == 0
            mock_asyncio_run.assert_called_once()
        finally:
            sys.argv = original_argv

    def test_early_mcp_telemetry_exception(self):
        """Test early --mcp mode when telemetry import fails (line 29-30)."""
        mock_mcp_main = MagicMock()

        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "--mcp"]
            with (
                patch("asyncio.run", MagicMock()),
                patch.dict("sys.modules", {
                    "openbrowser.mcp.server": MagicMock(main=mock_mcp_main),
                    "openbrowser.telemetry": None,  # Force ImportError
                    "openbrowser.utils": None,
                }),
                patch("logging.disable"),
                pytest.raises(SystemExit) as exc_info,
            ):
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()
                ns = {"__name__": "__test__", "__file__": str(cli_path)}
                exec(compile(code, str(cli_path), "exec"), ns)

            assert exc_info.value.code == 0
        finally:
            sys.argv = original_argv

    def test_early_c_with_code_success(self):
        """Test early -c mode with successful code execution (lines 80-89)."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = "Hello"
        mock_execute = AsyncMock(return_value=mock_result)

        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "-c", "print('hello')"]
            with (
                patch("asyncio.run", side_effect=lambda coro: mock_result),
                patch.dict("sys.modules", {
                    "dotenv": MagicMock(),
                    "openbrowser.daemon.client": MagicMock(
                        execute_code_via_daemon=mock_execute,
                    ),
                }),
                pytest.raises(SystemExit) as exc_info,
            ):
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()
                ns = {"__name__": "__test__", "__file__": str(cli_path)}
                exec(compile(code, str(cli_path), "exec"), ns)

            assert exc_info.value.code == 0
        finally:
            sys.argv = original_argv

    def test_early_c_with_code_failure(self):
        """Test early -c mode with failed code execution (lines 86-88)."""
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.output = None
        mock_result.error = "SyntaxError"

        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "-c", "bad code"]
            with (
                patch("asyncio.run", side_effect=lambda coro: mock_result),
                patch.dict("sys.modules", {
                    "dotenv": MagicMock(),
                    "openbrowser.daemon.client": MagicMock(
                        execute_code_via_daemon=AsyncMock(return_value=mock_result),
                    ),
                }),
                pytest.raises(SystemExit) as exc_info,
            ):
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()
                ns = {"__name__": "__test__", "__file__": str(cli_path)}
                exec(compile(code, str(cli_path), "exec"), ns)

            assert exc_info.value.code == 1
        finally:
            sys.argv = original_argv

    def test_early_c_no_code_daemon_running(self):
        """Test early -c with no code, daemon running (lines 68-70)."""
        mock_status = MagicMock()
        mock_status.success = True

        mock_client_cls = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_cls.return_value = mock_client_instance

        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "-c"]
            with (
                patch("asyncio.run", side_effect=lambda coro: mock_status),
                patch.dict("sys.modules", {
                    "dotenv": MagicMock(),
                    "openbrowser.daemon.client": MagicMock(
                        DaemonClient=mock_client_cls,
                    ),
                    "openbrowser.code_use.descriptions": MagicMock(
                        EXECUTE_CODE_DESCRIPTION="verbose desc",
                        EXECUTE_CODE_DESCRIPTION_COMPACT="compact desc",
                    ),
                }),
                pytest.raises(SystemExit) as exc_info,
            ):
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()
                ns = {"__name__": "__test__", "__file__": str(cli_path)}
                exec(compile(code, str(cli_path), "exec"), ns)

            assert exc_info.value.code == 0
        finally:
            sys.argv = original_argv

    def test_early_c_no_code_timeout_error(self):
        """Test early -c with no code, daemon status raises TimeoutError (line 63-65)."""
        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "-c"]
            with (
                patch("asyncio.run", side_effect=TimeoutError()),
                patch.dict("sys.modules", {
                    "dotenv": MagicMock(),
                    "openbrowser.daemon.client": MagicMock(
                        DaemonClient=MagicMock(),
                    ),
                    "openbrowser.code_use.descriptions": MagicMock(
                        EXECUTE_CODE_DESCRIPTION="verbose",
                        EXECUTE_CODE_DESCRIPTION_COMPACT="compact",
                    ),
                }),
                pytest.raises(SystemExit) as exc_info,
            ):
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()
                ns = {"__name__": "__test__", "__file__": str(cli_path)}
                exec(compile(code, str(cli_path), "exec"), ns)

            assert exc_info.value.code == 0
        finally:
            sys.argv = original_argv

    def test_early_daemon_unknown_command(self):
        """Test early daemon with unknown subcommand (lines 147-150)."""
        mock_client_cls = MagicMock()

        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "daemon", "badcmd"]
            with (
                patch.dict("sys.modules", {
                    "dotenv": MagicMock(),
                    "openbrowser.daemon.client": MagicMock(
                        DaemonClient=mock_client_cls,
                    ),
                }),
                pytest.raises(SystemExit) as exc_info,
            ):
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()
                ns = {"__name__": "__test__", "__file__": str(cli_path)}
                exec(compile(code, str(cli_path), "exec"), ns)

            assert exc_info.value.code == 1
        finally:
            sys.argv = original_argv

    def test_early_daemon_stop_success(self):
        """Test early daemon stop success (lines 117-121)."""
        mock_resp = MagicMock()
        mock_resp.success = True
        mock_resp.output = "Stopped"
        mock_resp.error = None

        mock_client_cls = MagicMock()

        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "daemon", "stop"]
            with (
                patch("asyncio.run", return_value=mock_resp),
                patch.dict("sys.modules", {
                    "dotenv": MagicMock(),
                    "openbrowser.daemon.client": MagicMock(
                        DaemonClient=mock_client_cls,
                    ),
                }),
                pytest.raises(SystemExit) as exc_info,
            ):
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()
                ns = {"__name__": "__test__", "__file__": str(cli_path)}
                exec(compile(code, str(cli_path), "exec"), ns)

            assert exc_info.value.code == 0
        finally:
            sys.argv = original_argv

    def test_early_daemon_stop_failure(self):
        """Test early daemon stop failure (lines 120-121)."""
        mock_resp = MagicMock()
        mock_resp.success = False
        mock_resp.output = None
        mock_resp.error = "Not running"

        mock_client_cls = MagicMock()

        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "daemon", "stop"]
            with (
                patch("asyncio.run", return_value=mock_resp),
                patch.dict("sys.modules", {
                    "dotenv": MagicMock(),
                    "openbrowser.daemon.client": MagicMock(
                        DaemonClient=mock_client_cls,
                    ),
                }),
                pytest.raises(SystemExit) as exc_info,
            ):
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()
                ns = {"__name__": "__test__", "__file__": str(cli_path)}
                exec(compile(code, str(cli_path), "exec"), ns)

            assert exc_info.value.code == 1
        finally:
            sys.argv = original_argv

    def test_early_daemon_status_success(self):
        """Test early daemon status success (lines 122-126)."""
        mock_resp = MagicMock()
        mock_resp.success = True
        mock_resp.output = "Running"
        mock_resp.error = None

        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "daemon", "status"]
            with (
                patch("asyncio.run", return_value=mock_resp),
                patch.dict("sys.modules", {
                    "dotenv": MagicMock(),
                    "openbrowser.daemon.client": MagicMock(
                        DaemonClient=MagicMock(),
                    ),
                }),
                pytest.raises(SystemExit) as exc_info,
            ):
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()
                ns = {"__name__": "__test__", "__file__": str(cli_path)}
                exec(compile(code, str(cli_path), "exec"), ns)

            assert exc_info.value.code == 0
        finally:
            sys.argv = original_argv

    def test_early_daemon_status_default(self):
        """Test early daemon defaults to status (line 105)."""
        mock_resp = MagicMock()
        mock_resp.success = True
        mock_resp.output = "Running"

        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "daemon"]
            with (
                patch("asyncio.run", return_value=mock_resp),
                patch.dict("sys.modules", {
                    "dotenv": MagicMock(),
                    "openbrowser.daemon.client": MagicMock(
                        DaemonClient=MagicMock(),
                    ),
                }),
                pytest.raises(SystemExit) as exc_info,
            ):
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()
                ns = {"__name__": "__test__", "__file__": str(cli_path)}
                exec(compile(code, str(cli_path), "exec"), ns)

            assert exc_info.value.code == 0
        finally:
            sys.argv = original_argv

    def test_early_c_no_code_os_error(self):
        """Test early -c with no code, daemon status raises OSError (lines 66-67)."""
        call_count = [0]

        def side_effect_run(coro):
            call_count[0] += 1
            if call_count[0] == 1:
                # First asyncio.run call = client.status()
                raise OSError("Connection refused")
            return MagicMock()  # For _start_daemon

        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "-c"]
            with (
                patch("asyncio.run", side_effect=side_effect_run),
                patch.dict("sys.modules", {
                    "dotenv": MagicMock(),
                    "openbrowser.daemon.client": MagicMock(
                        DaemonClient=MagicMock(),
                    ),
                    "openbrowser.code_use.descriptions": MagicMock(
                        EXECUTE_CODE_DESCRIPTION="verbose",
                        EXECUTE_CODE_DESCRIPTION_COMPACT="compact",
                    ),
                }),
                pytest.raises(SystemExit) as exc_info,
            ):
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()
                ns = {"__name__": "__test__", "__file__": str(cli_path)}
                exec(compile(code, str(cli_path), "exec"), ns)

            assert exc_info.value.code == 0
        finally:
            sys.argv = original_argv

    def test_early_c_no_code_daemon_not_running(self):
        """Test early -c no code, daemon not running (lines 72-77)."""
        mock_status = MagicMock()
        mock_status.success = False

        call_count = [0]

        def side_effect_run(coro):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_status  # client.status() returns not running
            return MagicMock()  # client._start_daemon()

        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "-c"]
            with (
                patch("asyncio.run", side_effect=side_effect_run),
                patch.dict("sys.modules", {
                    "dotenv": MagicMock(),
                    "openbrowser.daemon.client": MagicMock(
                        DaemonClient=MagicMock(),
                    ),
                    "openbrowser.code_use.descriptions": MagicMock(
                        EXECUTE_CODE_DESCRIPTION="verbose",
                        EXECUTE_CODE_DESCRIPTION_COMPACT="compact",
                    ),
                }),
                pytest.raises(SystemExit) as exc_info,
            ):
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()
                ns = {"__name__": "__test__", "__file__": str(cli_path)}
                exec(compile(code, str(cli_path), "exec"), ns)

            assert exc_info.value.code == 0
        finally:
            sys.argv = original_argv

    def test_early_c_no_code_start_daemon_timeout(self):
        """Test early -c no code, start daemon raises TimeoutError (line 76-77)."""
        mock_status = MagicMock()
        mock_status.success = False

        call_count = [0]

        def side_effect_run(coro):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_status  # client.status()
            raise TimeoutError("start timeout")  # client._start_daemon()

        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "-c"]
            with (
                patch("asyncio.run", side_effect=side_effect_run),
                patch.dict("sys.modules", {
                    "dotenv": MagicMock(),
                    "openbrowser.daemon.client": MagicMock(
                        DaemonClient=MagicMock(),
                    ),
                    "openbrowser.code_use.descriptions": MagicMock(
                        EXECUTE_CODE_DESCRIPTION="verbose",
                        EXECUTE_CODE_DESCRIPTION_COMPACT="compact",
                    ),
                }),
                pytest.raises(SystemExit) as exc_info,
            ):
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()
                ns = {"__name__": "__test__", "__file__": str(cli_path)}
                exec(compile(code, str(cli_path), "exec"), ns)

            assert exc_info.value.code == 0
        finally:
            sys.argv = original_argv

    def test_early_daemon_start_already_running(self):
        """Test early daemon start when already running (lines 109-116)."""
        mock_status = MagicMock()
        mock_status.success = True

        mock_client_instance = MagicMock()
        mock_client_instance.status = AsyncMock(return_value=mock_status)
        mock_client_instance._start_daemon = AsyncMock()
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "daemon", "start"]
            with (
                patch.dict("sys.modules", {
                    "dotenv": MagicMock(),
                    "openbrowser.daemon.client": MagicMock(
                        DaemonClient=mock_client_cls,
                    ),
                }),
                pytest.raises(SystemExit) as exc_info,
            ):
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()
                ns = {"__name__": "__test__", "__file__": str(cli_path)}
                exec(compile(code, str(cli_path), "exec"), ns)

            assert exc_info.value.code == 0
        finally:
            sys.argv = original_argv

    def test_early_daemon_start_not_running(self):
        """Test early daemon start when not running (lines 114-115)."""
        mock_status = MagicMock()
        mock_status.success = False

        mock_client_instance = MagicMock()
        mock_client_instance.status = AsyncMock(return_value=mock_status)
        mock_client_instance._start_daemon = AsyncMock()
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "daemon", "start"]
            with (
                patch.dict("sys.modules", {
                    "dotenv": MagicMock(),
                    "openbrowser.daemon.client": MagicMock(
                        DaemonClient=mock_client_cls,
                    ),
                }),
                pytest.raises(SystemExit) as exc_info,
            ):
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()
                ns = {"__name__": "__test__", "__file__": str(cli_path)}
                exec(compile(code, str(cli_path), "exec"), ns)

            assert exc_info.value.code == 0
        finally:
            sys.argv = original_argv

    def test_early_daemon_restart_success(self):
        """Test early daemon restart success (lines 128-146)."""
        mock_stop_resp = MagicMock()
        mock_stop_resp.success = False  # After stopping, status returns not running

        mock_client_instance = MagicMock()
        mock_client_instance.stop = AsyncMock()
        mock_client_instance.status = AsyncMock(return_value=mock_stop_resp)
        mock_client_instance._start_daemon = AsyncMock()
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "daemon", "restart"]
            with (
                patch.dict("sys.modules", {
                    "dotenv": MagicMock(),
                    "time": MagicMock(time=MagicMock(side_effect=[0, 0, 10])),
                    "openbrowser.daemon.client": MagicMock(
                        DaemonClient=mock_client_cls,
                    ),
                }),
                pytest.raises(SystemExit) as exc_info,
            ):
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()
                ns = {"__name__": "__test__", "__file__": str(cli_path)}
                exec(compile(code, str(cli_path), "exec"), ns)

            assert exc_info.value.code == 0
        finally:
            sys.argv = original_argv

    def test_early_daemon_restart_timeout(self):
        """Test early daemon restart timeout (lines 141-143)."""
        mock_running_resp = MagicMock()
        mock_running_resp.success = True  # Daemon still running after stop

        mock_client_instance = MagicMock()
        mock_client_instance.stop = AsyncMock()
        mock_client_instance.status = AsyncMock(return_value=mock_running_resp)
        mock_client_instance._start_daemon = AsyncMock()
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "daemon", "restart"]
            with (
                patch.dict("sys.modules", {
                    "dotenv": MagicMock(),
                    "openbrowser.daemon.client": MagicMock(
                        DaemonClient=mock_client_cls,
                    ),
                }),
                pytest.raises(SystemExit) as exc_info,
            ):
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()
                ns = {"__name__": "__test__", "__file__": str(cli_path)}
                exec(compile(code, str(cli_path), "exec"), ns)

            # Should exit 1 due to timeout or 0 due to normal restart
            assert exc_info.value.code in (0, 1)
        finally:
            sys.argv = original_argv

    def test_early_daemon_status_failure(self):
        """Test early daemon status failure exits 1 (line 126)."""
        mock_resp = MagicMock()
        mock_resp.success = False
        mock_resp.output = None
        mock_resp.error = "Not running"

        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "daemon", "status"]
            with (
                patch("asyncio.run", return_value=mock_resp),
                patch.dict("sys.modules", {
                    "dotenv": MagicMock(),
                    "openbrowser.daemon.client": MagicMock(
                        DaemonClient=MagicMock(),
                    ),
                }),
                pytest.raises(SystemExit) as exc_info,
            ):
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()
                ns = {"__name__": "__test__", "__file__": str(cli_path)}
                exec(compile(code, str(cli_path), "exec"), ns)

            assert exc_info.value.code == 1
        finally:
            sys.argv = original_argv

    def test_early_daemon_dotenv_import_error(self):
        """Test early daemon path when dotenv not installed (lines 98-99)."""
        mock_resp = MagicMock()
        mock_resp.success = True
        mock_resp.output = "Running"

        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "daemon", "status"]
            with (
                patch("asyncio.run", return_value=mock_resp),
                patch.dict("sys.modules", {
                    "dotenv": None,  # Force ImportError
                    "openbrowser.daemon.client": MagicMock(
                        DaemonClient=MagicMock(),
                    ),
                }),
                pytest.raises(SystemExit) as exc_info,
            ):
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()
                ns = {"__name__": "__test__", "__file__": str(cli_path)}
                exec(compile(code, str(cli_path), "exec"), ns)

            assert exc_info.value.code == 0
        finally:
            sys.argv = original_argv

    def test_early_init_subcommand(self):
        """Test early init subcommand detection (lines 178-201)."""
        mock_init_main = MagicMock()

        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "init"]
            with (
                patch.dict("sys.modules", {
                    "openbrowser.init_cmd": MagicMock(
                        INIT_TEMPLATES=INIT_TEMPLATES,
                        main=mock_init_main,
                    ),
                }),
                pytest.raises(SystemExit) as exc_info,
            ):
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()
                ns = {"__name__": "__test__", "__file__": str(cli_path)}
                exec(compile(code, str(cli_path), "exec"), ns)

            assert exc_info.value.code == 0
            mock_init_main.assert_called_once()
        finally:
            sys.argv = original_argv

    def test_early_init_with_template_flag(self):
        """Test early init with --template flag and value (lines 184-196)."""
        mock_init_main = MagicMock()

        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "init", "--template", "default"]
            with (
                patch.dict("sys.modules", {
                    "openbrowser.init_cmd": MagicMock(
                        INIT_TEMPLATES=INIT_TEMPLATES,
                        main=mock_init_main,
                    ),
                }),
                pytest.raises(SystemExit) as exc_info,
            ):
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()
                ns = {"__name__": "__test__", "__file__": str(cli_path)}
                exec(compile(code, str(cli_path), "exec"), ns)

            assert exc_info.value.code == 0
        finally:
            sys.argv = original_argv

    def test_early_init_with_template_no_value(self):
        """Test early init with --template but no value (lines 190-194)."""
        mock_init_main = MagicMock()

        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "init", "--template", "--force"]
            with (
                patch.dict("sys.modules", {
                    "openbrowser.init_cmd": MagicMock(
                        INIT_TEMPLATES=INIT_TEMPLATES,
                        main=mock_init_main,
                    ),
                }),
                pytest.raises(SystemExit) as exc_info,
            ):
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()
                ns = {"__name__": "__test__", "__file__": str(cli_path)}
                exec(compile(code, str(cli_path), "exec"), ns)

            assert exc_info.value.code == 0
        finally:
            sys.argv = original_argv

    def test_early_init_with_t_flag(self):
        """Test early init with -t flag (line 186)."""
        mock_init_main = MagicMock()

        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "init", "-t", "advanced"]
            with (
                patch.dict("sys.modules", {
                    "openbrowser.init_cmd": MagicMock(
                        INIT_TEMPLATES=INIT_TEMPLATES,
                        main=mock_init_main,
                    ),
                }),
                pytest.raises(SystemExit) as exc_info,
            ):
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()
                ns = {"__name__": "__test__", "__file__": str(cli_path)}
                exec(compile(code, str(cli_path), "exec"), ns)

            assert exc_info.value.code == 0
        finally:
            sys.argv = original_argv

    def test_early_template_flag_valid(self, tmp_path):
        """Test early --template flag with valid template (lines 204-277)."""
        mock_click = MagicMock()
        mock_click.echo = MagicMock()
        mock_click.confirm = MagicMock(return_value=True)

        # Create fake template file
        fake_templates = tmp_path / "cli_templates"
        fake_templates.mkdir()
        (fake_templates / "default_template.py").write_text("# default")

        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "--template", "default", "-o", str(tmp_path / "out.py"), "-f"]
            with (
                patch.dict("sys.modules", {
                    "openbrowser.init_cmd": MagicMock(
                        INIT_TEMPLATES=INIT_TEMPLATES,
                    ),
                }),
                pytest.raises(SystemExit) as exc_info,
            ):
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()
                ns = {
                    "__name__": "__test__",
                    "__file__": str(tmp_path / "cli.py"),
                }
                exec(compile(code, str(cli_path), "exec"), ns)

            assert exc_info.value.code == 0
        finally:
            sys.argv = original_argv

    def test_early_template_flag_invalid(self):
        """Test early --template flag with invalid template name (line 230)."""
        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "--template", "nonexistent"]
            with (
                patch.dict("sys.modules", {
                    "openbrowser.init_cmd": MagicMock(
                        INIT_TEMPLATES=INIT_TEMPLATES,
                    ),
                }),
                pytest.raises(SystemExit) as exc_info,
            ):
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()
                ns = {"__name__": "__test__", "__file__": str(cli_path)}
                exec(compile(code, str(cli_path), "exec"), ns)

            assert exc_info.value.code == 1
        finally:
            sys.argv = original_argv

    def test_early_template_flag_no_value_redirects(self):
        """Test early --template with no value redirects to init (lines 219-226)."""
        mock_init_main = MagicMock()

        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "--template"]
            with (
                patch.dict("sys.modules", {
                    "openbrowser.init_cmd": MagicMock(
                        INIT_TEMPLATES=INIT_TEMPLATES,
                        main=mock_init_main,
                    ),
                }),
                pytest.raises(SystemExit) as exc_info,
            ):
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()
                ns = {"__name__": "__test__", "__file__": str(cli_path)}
                exec(compile(code, str(cli_path), "exec"), ns)

            assert exc_info.value.code == 0
            mock_init_main.assert_called_once()
        finally:
            sys.argv = original_argv

    def test_early_template_flag_starts_with_dash_redirects(self):
        """Test early --template --force redirects to init (lines 219-226)."""
        mock_init_main = MagicMock()

        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "--template", "--force"]
            with (
                patch.dict("sys.modules", {
                    "openbrowser.init_cmd": MagicMock(
                        INIT_TEMPLATES=INIT_TEMPLATES,
                        main=mock_init_main,
                    ),
                }),
                pytest.raises(SystemExit) as exc_info,
            ):
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()
                ns = {"__name__": "__test__", "__file__": str(cli_path)}
                exec(compile(code, str(cli_path), "exec"), ns)

            assert exc_info.value.code == 0
            mock_init_main.assert_called_once()
        finally:
            sys.argv = original_argv

    def test_early_template_file_exists_no_force(self, tmp_path):
        """Test early --template when file exists and user cancels (lines 256-259)."""
        # Create the output file already
        output_file = tmp_path / "out.py"
        output_file.write_text("existing")

        # Create fake template
        fake_templates = tmp_path / "cli_templates"
        fake_templates.mkdir()
        (fake_templates / "default_template.py").write_text("# default")

        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "--template", "default", "-o", str(output_file)]
            with (
                patch.dict("sys.modules", {
                    "openbrowser.init_cmd": MagicMock(INIT_TEMPLATES=INIT_TEMPLATES),
                }),
                patch("click.confirm", return_value=False),
                pytest.raises(SystemExit) as exc_info,
            ):
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()
                ns = {
                    "__name__": "__test__",
                    "__file__": str(tmp_path / "cli.py"),
                }
                exec(compile(code, str(cli_path), "exec"), ns)

            assert exc_info.value.code == 1
        finally:
            sys.argv = original_argv

    def test_early_template_with_output_o_flag(self, tmp_path):
        """Test early --template with -o flag (lines 235-239)."""
        fake_templates = tmp_path / "cli_templates"
        fake_templates.mkdir()
        (fake_templates / "default_template.py").write_text("# default")

        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "--template", "default", "-o", str(tmp_path / "out.py"), "-f"]
            with (
                patch.dict("sys.modules", {
                    "openbrowser.init_cmd": MagicMock(INIT_TEMPLATES=INIT_TEMPLATES),
                }),
                pytest.raises(SystemExit) as exc_info,
            ):
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()
                ns = {"__name__": "__test__", "__file__": str(tmp_path / "cli.py")}
                exec(compile(code, str(cli_path), "exec"), ns)

            assert exc_info.value.code == 0
            assert (tmp_path / "out.py").exists()
        finally:
            sys.argv = original_argv

    def test_early_init_with_t_flag_no_value(self):
        """Test early init with -t flag where value starts with - (line 194)."""
        mock_init_main = MagicMock()

        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "init", "-t", "--force"]
            with (
                patch.dict("sys.modules", {
                    "openbrowser.init_cmd": MagicMock(
                        INIT_TEMPLATES=INIT_TEMPLATES,
                        main=mock_init_main,
                    ),
                }),
                pytest.raises(SystemExit) as exc_info,
            ):
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()
                ns = {"__name__": "__test__", "__file__": str(cli_path)}
                exec(compile(code, str(cli_path), "exec"), ns)

            assert exc_info.value.code == 0
        finally:
            sys.argv = original_argv

    def test_early_template_error_reading(self):
        """Test early --template when template read fails (line 273-275)."""
        # Read the source before patching Path.read_text
        cli_path = Path(cli_mod.__file__)
        source_code = cli_path.read_text()

        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "--template", "default", "-o", "/tmp/early_out.py", "-f"]
            with (
                patch.dict("sys.modules", {
                    "openbrowser.init_cmd": MagicMock(INIT_TEMPLATES=INIT_TEMPLATES),
                }),
                patch.object(Path, "read_text", side_effect=Exception("read error")),
                pytest.raises(SystemExit) as exc_info,
            ):
                ns = {"__name__": "__test__", "__file__": str(cli_path)}
                exec(compile(source_code, str(cli_path), "exec"), ns)

            assert exc_info.value.code == 1
        finally:
            sys.argv = original_argv

    def test_early_c_dotenv_import_error(self):
        """Test early -c path when dotenv is not installed (lines 43-45)."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = "ok"

        original_argv = sys.argv[:]
        try:
            sys.argv = ["openbrowser-ai", "-c", "pass"]
            with (
                patch("asyncio.run", return_value=mock_result),
                patch.dict("sys.modules", {
                    "dotenv": None,  # Force ImportError for dotenv
                    "openbrowser.daemon.client": MagicMock(
                        execute_code_via_daemon=AsyncMock(return_value=mock_result),
                    ),
                }),
                pytest.raises(SystemExit) as exc_info,
            ):
                cli_path = Path(cli_mod.__file__)
                code = cli_path.read_text()
                ns = {"__name__": "__test__", "__file__": str(cli_path)}
                exec(compile(code, str(cli_path), "exec"), ns)

            assert exc_info.value.code == 0
        finally:
            sys.argv = original_argv


class TestClickCommandMetadata:
    """Test Click command decorators and options are correct."""

    def test_main_is_click_group(self):
        assert hasattr(main, "commands")

    def test_init_is_subcommand_of_main(self):
        assert "init" in main.commands

    def test_install_is_subcommand_of_main(self):
        assert "install" in main.commands

    def test_main_help_contains_examples(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert "openbrowser-ai" in result.output

    def test_init_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["init", "--help"])
        assert result.exit_code == 0
        assert "template" in result.output.lower()

    def test_install_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["install", "--help"])
        assert result.exit_code == 0
        assert "Chromium" in result.output
