"""Comprehensive tests for openbrowser.utils (utils.py) module.

Covers: SignalHandler, time_execution_sync, time_execution_async, singleton,
check_env_variables, is_unsafe_pattern, is_new_tab_page, match_url_with_domain_pattern,
merge_dicts, get_openbrowser_version, check_latest_openbrowser_version, get_git_info,
_log_pretty_path, _log_pretty_url.
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openbrowser.utils import (
    SignalHandler,
    _log_pretty_path,
    _log_pretty_url,
    check_env_variables,
    check_latest_openbrowser_version,
    get_git_info,
    get_openbrowser_version,
    is_new_tab_page,
    is_unsafe_pattern,
    match_url_with_domain_pattern,
    merge_dicts,
    singleton,
    time_execution_async,
    time_execution_sync,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# _log_pretty_path
# ---------------------------------------------------------------------------


class TestLogPrettyPath:
    def test_none_returns_empty(self):
        assert _log_pretty_path(None) == ""

    def test_empty_string_returns_empty(self):
        assert _log_pretty_path("") == ""

    def test_whitespace_only_returns_empty(self):
        assert _log_pretty_path("   ") == ""

    def test_home_dir_replaced(self):
        home = str(Path.home())
        result = _log_pretty_path(f"{home}/some/path")
        assert result.startswith("~")

    def test_path_with_spaces_quoted(self):
        result = _log_pretty_path("/some/path with spaces/file.txt")
        assert result.startswith('"')
        assert result.endswith('"')

    def test_path_without_spaces_not_quoted(self):
        result = _log_pretty_path("/some/path/file.txt")
        assert not result.startswith('"')

    def test_non_path_type(self):
        result = _log_pretty_path(42)
        assert "<int>" in result

    def test_dict_type(self):
        result = _log_pretty_path({"key": "value"})
        assert "<dict>" in result

    def test_pathlib_path(self):
        result = _log_pretty_path(Path("/tmp/test.txt"))
        assert "test.txt" in result


# ---------------------------------------------------------------------------
# _log_pretty_url
# ---------------------------------------------------------------------------


class TestLogPrettyUrl:
    def test_strips_protocol(self):
        result = _log_pretty_url("https://example.com/page")
        assert "https://" not in result

    def test_strips_www(self):
        result = _log_pretty_url("https://www.example.com")
        assert "www." not in result

    def test_truncates_long_url(self):
        result = _log_pretty_url("https://example.com/very/long/path/that/exceeds/limit", max_len=22)
        assert result.endswith("...")
        assert len(result) == 25  # 22 + 3 for ...

    def test_no_truncation_when_short(self):
        result = _log_pretty_url("https://a.com", max_len=50)
        assert "..." not in result

    def test_no_max_len(self):
        long_url = "https://example.com/" + "a" * 100
        result = _log_pretty_url(long_url, max_len=None)
        assert "..." not in result

    def test_http_stripped(self):
        result = _log_pretty_url("http://example.com")
        assert "http://" not in result


# ---------------------------------------------------------------------------
# singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_creates_only_one_instance(self):
        @singleton
        class MyClass:
            def __init__(self, value):
                self.value = value

        obj1 = MyClass(1)
        obj2 = MyClass(2)
        assert obj1 is obj2
        assert obj1.value == 1  # Second call doesn't create new instance


# ---------------------------------------------------------------------------
# check_env_variables
# ---------------------------------------------------------------------------


class TestCheckEnvVariables:
    def test_all_set(self):
        with patch.dict(os.environ, {"A": "1", "B": "2"}):
            assert check_env_variables(["A", "B"]) is True

    def test_one_missing(self):
        with patch.dict(os.environ, {"A": "1"}, clear=False):
            os.environ.pop("B", None)
            assert check_env_variables(["A", "B"]) is False

    def test_any_mode(self):
        with patch.dict(os.environ, {"A": "1"}, clear=False):
            os.environ.pop("B", None)
            assert check_env_variables(["A", "B"], any_or_all=any) is True

    def test_all_empty(self):
        with patch.dict(os.environ, {"A": "", "B": ""}, clear=False):
            assert check_env_variables(["A", "B"]) is False

    def test_whitespace_only(self):
        with patch.dict(os.environ, {"A": "  "}, clear=False):
            # strip() returns empty string, which is falsy
            assert check_env_variables(["A"]) is False


# ---------------------------------------------------------------------------
# is_unsafe_pattern
# ---------------------------------------------------------------------------


class TestIsUnsafePattern:
    def test_safe_wildcard(self):
        assert is_unsafe_pattern("*.example.com") is False

    def test_safe_tld_wildcard(self):
        assert is_unsafe_pattern("example.*") is False

    def test_embedded_wildcard(self):
        assert is_unsafe_pattern("exa*mple.com") is True

    def test_no_wildcard(self):
        assert is_unsafe_pattern("example.com") is False

    def test_with_scheme(self):
        assert is_unsafe_pattern("https://exa*mple.com") is True

    def test_with_scheme_safe(self):
        assert is_unsafe_pattern("https://example.com") is False

    def test_double_wildcard_is_safe(self):
        # *.domain.* after removal: "domain" - no * left
        # Actually *.domain.* -> bare_domain = "domain" -> no * left
        assert is_unsafe_pattern("*.domain.*") is False


# ---------------------------------------------------------------------------
# is_new_tab_page
# ---------------------------------------------------------------------------


class TestIsNewTabPage:
    def test_about_blank(self):
        assert is_new_tab_page("about:blank") is True

    def test_chrome_new_tab_with_slash(self):
        assert is_new_tab_page("chrome://new-tab-page/") is True

    def test_chrome_new_tab_without_slash(self):
        assert is_new_tab_page("chrome://new-tab-page") is True

    def test_chrome_newtab_with_slash(self):
        assert is_new_tab_page("chrome://newtab/") is True

    def test_chrome_newtab_without_slash(self):
        assert is_new_tab_page("chrome://newtab") is True

    def test_regular_url(self):
        assert is_new_tab_page("https://google.com") is False

    def test_empty_string(self):
        assert is_new_tab_page("") is False


# ---------------------------------------------------------------------------
# match_url_with_domain_pattern
# ---------------------------------------------------------------------------


class TestMatchUrlWithDomainPattern:
    def test_exact_match(self):
        assert match_url_with_domain_pattern("https://example.com", "example.com") is True

    def test_wildcard_subdomain(self):
        assert match_url_with_domain_pattern("https://sub.example.com", "*.example.com") is True

    def test_wildcard_matches_bare_domain(self):
        assert match_url_with_domain_pattern("https://example.com", "*.example.com") is True

    def test_wildcard_star_only(self):
        assert match_url_with_domain_pattern("https://anything.com", "*") is True

    def test_scheme_mismatch(self):
        assert match_url_with_domain_pattern("http://example.com", "example.com") is False

    def test_scheme_wildcard(self):
        assert match_url_with_domain_pattern("http://example.com", "http*://example.com") is True

    def test_new_tab_page_always_false(self):
        assert match_url_with_domain_pattern("about:blank", "about:blank") is False

    def test_no_hostname(self):
        assert match_url_with_domain_pattern("data:text/html,test", "example.com") is False

    def test_multiple_wildcards_rejected(self):
        assert match_url_with_domain_pattern("https://sub.example.com", "*.*.example.com") is False

    def test_tld_wildcard_rejected(self):
        assert match_url_with_domain_pattern("https://example.com", "example.*") is False

    def test_embedded_wildcard_rejected(self):
        assert match_url_with_domain_pattern("https://google.com", "goo*le.com") is False

    def test_chrome_extension_wildcard(self):
        assert match_url_with_domain_pattern(
            "chrome-extension://abcdefg", "chrome-extension://*"
        ) is True

    def test_port_in_pattern_stripped(self):
        assert match_url_with_domain_pattern("https://example.com:8080", "example.com:8080") is True

    def test_case_insensitive(self):
        assert match_url_with_domain_pattern("https://Example.COM", "example.com") is True

    def test_exception_returns_false(self):
        # Pass something that would cause an error in urlparse processing
        assert match_url_with_domain_pattern("", "example.com") is False

    def test_http_scheme_explicit(self):
        assert match_url_with_domain_pattern("http://example.com", "http://example.com") is True

    def test_default_scheme_is_https(self):
        assert match_url_with_domain_pattern("https://example.com", "example.com") is True
        assert match_url_with_domain_pattern("http://example.com", "example.com") is False


# ---------------------------------------------------------------------------
# merge_dicts
# ---------------------------------------------------------------------------


class TestMergeDicts:
    def test_simple_merge(self):
        a = {"x": 1}
        b = {"y": 2}
        result = merge_dicts(a, b)
        assert result == {"x": 1, "y": 2}

    def test_nested_merge(self):
        a = {"config": {"a": 1}}
        b = {"config": {"b": 2}}
        result = merge_dicts(a, b)
        assert result == {"config": {"a": 1, "b": 2}}

    def test_list_concatenation(self):
        a = {"items": [1, 2]}
        b = {"items": [3, 4]}
        result = merge_dicts(a, b)
        assert result == {"items": [1, 2, 3, 4]}

    def test_conflict_raises(self):
        a = {"x": 1}
        b = {"x": 2}
        with pytest.raises(Exception, match="Conflict"):
            merge_dicts(a, b)

    def test_same_values_no_conflict(self):
        a = {"x": 1}
        b = {"x": 1}
        result = merge_dicts(a, b)
        assert result == {"x": 1}


# ---------------------------------------------------------------------------
# get_openbrowser_version
# ---------------------------------------------------------------------------


class TestGetOpenbrowserVersion:
    def test_returns_string(self):
        # Clear cache so it's freshly computed
        get_openbrowser_version.cache_clear()
        version = get_openbrowser_version()
        assert isinstance(version, str)
        assert version != ""

    def test_caching(self):
        get_openbrowser_version.cache_clear()
        v1 = get_openbrowser_version()
        v2 = get_openbrowser_version()
        assert v1 == v2


# ---------------------------------------------------------------------------
# check_latest_openbrowser_version
# ---------------------------------------------------------------------------


class TestCheckLatestVersion:
    @pytest.mark.asyncio
    async def test_returns_string_or_none(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"info": {"version": "0.1.37"}}

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await check_latest_openbrowser_version()
            assert isinstance(result, str)
            assert result == "0.1.37"


# ---------------------------------------------------------------------------
# get_git_info
# ---------------------------------------------------------------------------


class TestGetGitInfo:
    def test_returns_dict_or_none(self):
        get_git_info.cache_clear()
        result = get_git_info()
        if result is not None:
            assert "commit_hash" in result
            assert "branch" in result
        # None is also valid if not in a git repo


# ---------------------------------------------------------------------------
# time_execution_sync
# ---------------------------------------------------------------------------


class TestTimeExecutionSync:
    def test_decorator_preserves_return(self):
        @time_execution_sync("test")
        def my_func(x):
            return x * 2

        assert my_func(5) == 10

    def test_decorator_preserves_function_name(self):
        @time_execution_sync("test")
        def my_func():
            pass

        assert my_func.__name__ == "my_func"


# ---------------------------------------------------------------------------
# time_execution_async
# ---------------------------------------------------------------------------


class TestTimeExecutionAsync:
    @pytest.mark.asyncio
    async def test_decorator_preserves_return(self):
        @time_execution_async("test")
        async def my_func(x):
            return x * 2

        assert await my_func(5) == 10

    @pytest.mark.asyncio
    async def test_decorator_preserves_function_name(self):
        @time_execution_async("test")
        async def my_func():
            pass

        assert my_func.__name__ == "my_func"


# ---------------------------------------------------------------------------
# SignalHandler basic tests (no actual signal sending)
# ---------------------------------------------------------------------------


class TestSignalHandler:
    def test_init_defaults(self):
        handler = SignalHandler(exit_on_second_int=True)
        assert handler._exit_on_second_int is True
        assert handler._sigint_count == 0
        assert handler._is_paused is False
        assert handler._registered is False

    def test_init_custom_exit_on_second_int(self):
        handler = SignalHandler(exit_on_second_int=False)
        assert handler._exit_on_second_int is False

    def test_reset(self):
        handler = SignalHandler()
        handler._sigint_count = 3
        handler._is_paused = True
        handler.reset()
        assert handler._sigint_count == 0
        assert handler._is_paused is False

    def test_is_paused_property(self):
        handler = SignalHandler()
        assert handler.is_paused is False
        handler._is_paused = True
        assert handler.is_paused is True

    def test_interrupt_count_property(self):
        handler = SignalHandler()
        assert handler.interrupt_count == 0
        handler._sigint_count = 2
        assert handler.interrupt_count == 2

    def test_register_and_unregister(self):
        handler = SignalHandler()
        handler.register()
        assert handler._registered is True
        handler.unregister()
        assert handler._registered is False

    def test_context_manager(self):
        handler = SignalHandler()
        with handler as h:
            assert h is handler
            assert h._registered is True
        assert handler._registered is False
