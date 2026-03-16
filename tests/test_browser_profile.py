"""Comprehensive tests for openbrowser.browser.profile module.

Covers: ViewportSize, get_window_adjustments, validate_url, validate_float_range,
validate_cli_arg, BrowserChannel, BrowserContextArgs, BrowserLaunchArgs,
BrowserConnectArgs, BrowserLaunchPersistentContextArgs, ProxySettings,
BrowserProfile and its validators.
"""

import logging
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from openbrowser.browser.profile import (
    CHROME_DEBUG_PORT,
    CHROME_DEFAULT_ARGS,
    CHROME_DISABLE_SECURITY_ARGS,
    CHROME_DOCKER_ARGS,
    CHROME_HEADLESS_ARGS,
    DOMAIN_OPTIMIZATION_THRESHOLD,
    BrowserChannel,
    BrowserConnectArgs,
    BrowserContextArgs,
    BrowserLaunchArgs,
    BrowserLaunchPersistentContextArgs,
    BrowserNewContextArgs,
    BrowserProfile,
    ProxySettings,
    RecordHarContent,
    RecordHarMode,
    ViewportSize,
    get_window_adjustments,
    validate_cli_arg,
    validate_float_range,
    validate_url,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ViewportSize
# ---------------------------------------------------------------------------


class TestViewportSize:
    def test_creation(self):
        vs = ViewportSize(width=1920, height=1080)
        assert vs.width == 1920
        assert vs.height == 1080

    def test_getitem(self):
        vs = ViewportSize(width=1920, height=1080)
        assert vs["width"] == 1920
        assert vs["height"] == 1080

    def test_setitem(self):
        vs = ViewportSize(width=0, height=0)
        vs["width"] = 800
        vs["height"] = 600
        assert vs.width == 800
        assert vs.height == 600

    def test_negative_width_rejected(self):
        with pytest.raises(Exception):
            ViewportSize(width=-1, height=100)

    def test_zero_width_allowed(self):
        vs = ViewportSize(width=0, height=0)
        assert vs.width == 0


# ---------------------------------------------------------------------------
# get_window_adjustments
# ---------------------------------------------------------------------------


class TestGetWindowAdjustments:
    def test_returns_tuple(self):
        result = get_window_adjustments()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_macos(self):
        with patch.object(sys, "platform", "darwin"):
            x, y = get_window_adjustments()
            assert x == -4
            assert y == 24

    def test_windows(self):
        with patch.object(sys, "platform", "win32"):
            x, y = get_window_adjustments()
            assert x == -8
            assert y == 0

    def test_linux(self):
        with patch.object(sys, "platform", "linux"):
            x, y = get_window_adjustments()
            assert x == 0
            assert y == 0


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


class TestValidateUrl:
    def test_valid_url(self):
        result = validate_url("https://example.com")
        assert result == "https://example.com"

    def test_invalid_url_no_netloc(self):
        with pytest.raises(ValueError, match="Invalid URL"):
            validate_url("not-a-url")

    def test_valid_url_with_scheme_check(self):
        result = validate_url("https://example.com", schemes=["https"])
        assert result == "https://example.com"

    def test_invalid_scheme(self):
        with pytest.raises(ValueError, match="invalid scheme"):
            validate_url("ftp://example.com", schemes=["http", "https"])

    def test_no_scheme_check(self):
        result = validate_url("http://example.com")
        assert result == "http://example.com"


class TestValidateFloatRange:
    def test_in_range(self):
        assert validate_float_range(5.0, 0.0, 10.0) == 5.0

    def test_at_min(self):
        assert validate_float_range(0.0, 0.0, 10.0) == 0.0

    def test_at_max(self):
        assert validate_float_range(10.0, 0.0, 10.0) == 10.0

    def test_below_min(self):
        with pytest.raises(ValueError, match="outside of range"):
            validate_float_range(-1.0, 0.0, 10.0)

    def test_above_max(self):
        with pytest.raises(ValueError, match="outside of range"):
            validate_float_range(11.0, 0.0, 10.0)


class TestValidateCliArg:
    def test_valid_arg(self):
        assert validate_cli_arg("--headless") == "--headless"

    def test_valid_arg_with_value(self):
        assert validate_cli_arg("--user-data-dir=/tmp/test") == "--user-data-dir=/tmp/test"

    def test_invalid_arg(self):
        with pytest.raises(ValueError, match="Invalid CLI argument"):
            validate_cli_arg("headless")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestEnums:
    def test_record_har_content_values(self):
        assert RecordHarContent.OMIT == "omit"
        assert RecordHarContent.EMBED == "embed"
        assert RecordHarContent.ATTACH == "attach"

    def test_record_har_mode_values(self):
        assert RecordHarMode.FULL == "full"
        assert RecordHarMode.MINIMAL == "minimal"

    def test_browser_channel_values(self):
        assert BrowserChannel.CHROMIUM == "chromium"
        assert BrowserChannel.CHROME == "chrome"
        assert BrowserChannel.MSEDGE == "msedge"


# ---------------------------------------------------------------------------
# BrowserContextArgs
# ---------------------------------------------------------------------------


class TestBrowserContextArgs:
    def test_defaults(self):
        ctx = BrowserContextArgs()
        assert ctx.accept_downloads is True
        assert "clipboardReadWrite" in ctx.permissions
        assert ctx.viewport is None
        assert ctx.user_agent is None

    def test_custom_permissions(self):
        ctx = BrowserContextArgs(permissions=["camera"])
        assert ctx.permissions == ["camera"]


# ---------------------------------------------------------------------------
# BrowserConnectArgs
# ---------------------------------------------------------------------------


class TestBrowserConnectArgs:
    def test_defaults(self):
        args = BrowserConnectArgs()
        assert args.headers is None

    def test_custom_headers(self):
        args = BrowserConnectArgs(headers={"Authorization": "Bearer token"})
        assert args.headers["Authorization"] == "Bearer token"


# ---------------------------------------------------------------------------
# BrowserLaunchArgs
# ---------------------------------------------------------------------------


class TestBrowserLaunchArgs:
    def test_defaults(self):
        args = BrowserLaunchArgs()
        assert args.headless is None
        assert args.executable_path is None
        assert args.devtools is False

    def test_args_as_dict(self):
        result = BrowserLaunchArgs.args_as_dict(["--headless", "--user-data-dir=/tmp/test"])
        assert "headless" in result
        assert result["user-data-dir"] == "/tmp/test"

    def test_args_as_list(self):
        result = BrowserLaunchArgs.args_as_list({"headless": "", "user-data-dir": "/tmp"})
        assert "--headless" in result
        assert "--user-data-dir=/tmp" in result

    def test_args_as_dict_with_dashes(self):
        result = BrowserLaunchArgs.args_as_dict(["--some-key=value"])
        assert "some-key" in result
        assert result["some-key"] == "value"

    def test_headless_devtools_conflict(self):
        with pytest.raises(Exception):
            BrowserLaunchArgs(headless=True, devtools=True)

    def test_custom_args(self):
        args = BrowserLaunchArgs(args=["--custom-flag=value"])
        assert "--custom-flag=value" in args.args

    def test_downloads_path_auto_generated(self):
        args = BrowserLaunchArgs()
        assert args.downloads_path is not None


# ---------------------------------------------------------------------------
# BrowserLaunchPersistentContextArgs
# ---------------------------------------------------------------------------


class TestBrowserLaunchPersistentContextArgs:
    def test_user_data_dir_none_creates_temp(self):
        args = BrowserLaunchPersistentContextArgs(user_data_dir=None)
        assert args.user_data_dir is not None
        assert "openbrowser-user-data-dir" in str(args.user_data_dir)

    def test_user_data_dir_expanded(self, tmp_path):
        args = BrowserLaunchPersistentContextArgs(user_data_dir=str(tmp_path))
        assert Path(args.user_data_dir).is_absolute()


# ---------------------------------------------------------------------------
# ProxySettings
# ---------------------------------------------------------------------------


class TestProxySettings:
    def test_defaults(self):
        proxy = ProxySettings()
        assert proxy.server is None
        assert proxy.bypass is None
        assert proxy.username is None
        assert proxy.password is None

    def test_custom(self):
        proxy = ProxySettings(
            server="http://proxy:8080",
            bypass="localhost,127.0.0.1",
            username="user",
            password="pass",
        )
        assert proxy.server == "http://proxy:8080"

    def test_getitem(self):
        proxy = ProxySettings(server="http://proxy:8080")
        assert proxy["server"] == "http://proxy:8080"
        assert proxy["bypass"] is None


# ---------------------------------------------------------------------------
# BrowserProfile
# ---------------------------------------------------------------------------


class TestBrowserProfile:
    def test_default_creation(self):
        profile = BrowserProfile()
        assert profile.cdp_url is None
        assert profile.disable_security is False
        assert profile.deterministic_rendering is False
        assert profile.allowed_domains is None
        assert profile.prohibited_domains is None

    def test_str(self):
        profile = BrowserProfile()
        assert str(profile) == "BrowserProfile"

    def test_repr(self, tmp_path):
        profile = BrowserProfile(user_data_dir=str(tmp_path / "data"))
        result = repr(profile)
        assert "BrowserProfile" in result


    def test_disable_security(self, tmp_path):
        profile = BrowserProfile(disable_security=True, enable_default_extensions=False, user_data_dir=str(tmp_path / "data"))
        args = profile.get_args()
        assert any("--disable-web-security" in arg for arg in args)

    def test_headless_mode(self, tmp_path):
        profile = BrowserProfile(headless=True, enable_default_extensions=False, user_data_dir=str(tmp_path / "data"))
        args = profile.get_args()
        assert any("--headless" in arg for arg in args)

    def test_custom_window_size(self, tmp_path):
        profile = BrowserProfile(window_size=ViewportSize(width=1920, height=1080), headless=False, enable_default_extensions=False, user_data_dir=str(tmp_path / "data"))
        args = profile.get_args()
        assert any("--window-size=1920,1080" in arg for arg in args)

    def test_proxy_settings_in_args(self, tmp_path):
        profile = BrowserProfile(
            proxy=ProxySettings(server="http://proxy:8080", bypass="localhost"),
            enable_default_extensions=False,
            user_data_dir=str(tmp_path / "data"),
        )
        args = profile.get_args()
        assert any("--proxy-server" in arg for arg in args)
        assert any("--proxy-bypass-list" in arg for arg in args)

    def test_user_agent_in_args(self, tmp_path):
        profile = BrowserProfile(user_agent="Custom Agent/1.0", enable_default_extensions=False, user_data_dir=str(tmp_path / "data"))
        args = profile.get_args()
        assert any("--user-agent=Custom Agent/1.0" in arg for arg in args)


    def test_allowed_domains_list(self):
        profile = BrowserProfile(allowed_domains=["example.com", "*.google.com"])
        assert isinstance(profile.allowed_domains, list)
        assert len(profile.allowed_domains) == 2

    def test_large_domain_list_optimized_to_set(self):
        domains = [f"domain{i}.com" for i in range(150)]
        profile = BrowserProfile(allowed_domains=domains)
        assert isinstance(profile.allowed_domains, set)

    def test_prohibited_domains(self):
        profile = BrowserProfile(prohibited_domains=["evil.com"])
        assert profile.prohibited_domains == ["evil.com"]

    def test_block_ip_addresses(self):
        profile = BrowserProfile(block_ip_addresses=True)
        assert profile.block_ip_addresses is True

    def test_highlight_conflict_resolved(self):
        """When both highlight_elements and dom_highlight_elements are True,
        highlight_elements should be set to False."""
        profile = BrowserProfile(highlight_elements=True, dom_highlight_elements=True)
        assert profile.highlight_elements is False
        assert profile.dom_highlight_elements is True

    def test_deterministic_rendering_warning(self, tmp_path):
        """deterministic_rendering=True should not crash."""
        profile = BrowserProfile(deterministic_rendering=True, user_data_dir=str(tmp_path / "data"))
        assert profile.deterministic_rendering is True
        args = profile.get_args()
        assert any("--deterministic-mode" in arg for arg in args)


    def test_proxy_bypass_without_server_warning(self):
        """proxy.bypass without proxy.server should not crash."""
        profile = BrowserProfile(proxy=ProxySettings(bypass="localhost"))
        assert profile.proxy.bypass == "localhost"

    def test_window_position(self, tmp_path):
        profile = BrowserProfile(window_position=ViewportSize(width=100, height=200), headless=False, user_data_dir=str(tmp_path / "data"))
        args = profile.get_args()
        assert any("--window-position=100,200" in arg for arg in args)

    def test_cookie_whitelist_domains_default(self):
        profile = BrowserProfile()
        assert "nature.com" in profile.cookie_whitelist_domains


    def test_get_args_dedupes(self, tmp_path):
        """get_args should deduplicate args."""
        profile = BrowserProfile(args=["--custom-flag"], user_data_dir=str(tmp_path / "data"))
        args = profile.get_args()
        # No duplicate --custom-flag
        custom_count = sum(1 for a in args if "custom-flag" in a)
        assert custom_count == 1


    def test_ignore_default_args_true(self, tmp_path):
        """Setting ignore_default_args=True should exclude all defaults."""
        profile = BrowserProfile(ignore_default_args=True, user_data_dir=str(tmp_path / "data"))
        args = profile.get_args()
        # Should still include user_data_dir at minimum
        assert any("--user-data-dir" in a for a in args)


    def test_domain_list_none_stays_none(self):
        profile = BrowserProfile(allowed_domains=None)
        assert profile.allowed_domains is None

    def test_domain_set_stays_set(self):
        profile = BrowserProfile(allowed_domains={"example.com", "test.com"})
        assert isinstance(profile.allowed_domains, set)

    def test_cross_origin_iframes_default(self):
        profile = BrowserProfile()
        assert profile.cross_origin_iframes is True

    def test_max_iframes_default(self):
        profile = BrowserProfile()
        assert profile.max_iframes == 100

    def test_max_iframe_depth_default(self):
        profile = BrowserProfile()
        assert profile.max_iframe_depth == 5

    def test_auto_download_pdfs_default(self):
        profile = BrowserProfile()
        assert profile.auto_download_pdfs is True


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_chrome_debug_port(self):
        assert CHROME_DEBUG_PORT == 9242

    def test_domain_optimization_threshold(self):
        assert DOMAIN_OPTIMIZATION_THRESHOLD == 100

    def test_chrome_default_args_is_list(self):
        assert isinstance(CHROME_DEFAULT_ARGS, list)
        assert len(CHROME_DEFAULT_ARGS) > 0

    def test_chrome_headless_args(self):
        assert "--headless=new" in CHROME_HEADLESS_ARGS

    def test_chrome_docker_args(self):
        assert "--no-sandbox" in CHROME_DOCKER_ARGS

    def test_chrome_disable_security_args(self):
        assert "--disable-web-security" in CHROME_DISABLE_SECURITY_ARGS


# ---------------------------------------------------------------------------
# BrowserNewContextArgs
# ---------------------------------------------------------------------------


class TestBrowserNewContextArgs:
    def test_defaults(self):
        args = BrowserNewContextArgs()
        assert args.storage_state is None

    def test_storage_state_string(self):
        args = BrowserNewContextArgs(storage_state="/path/to/state.json")
        assert args.storage_state == "/path/to/state.json"

    def test_storage_state_dict(self):
        state = {"cookies": [], "origins": []}
        args = BrowserNewContextArgs(storage_state=state)
        assert args.storage_state == state
