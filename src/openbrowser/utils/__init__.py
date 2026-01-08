"""Utility modules for openbrowser-ai."""

import logging

from openbrowser.utils.signal_handler import SignalHandler, AsyncSignalHandler

# Re-export from the main utils module (openbrowser/utils.py)
# These are imported via: from openbrowser.utils import _log_pretty_path, logger
# We need to import from the parent module's utils.py file
import sys
from pathlib import Path

# Import from the utils.py file in the parent directory
_parent_utils = sys.modules.get('openbrowser.utils_module')
if _parent_utils is None:
    import importlib.util
    _utils_path = Path(__file__).parent.parent / 'utils.py'
    if _utils_path.exists():
        _spec = importlib.util.spec_from_file_location('openbrowser.utils_module', _utils_path)
        _parent_utils = importlib.util.module_from_spec(_spec)
        sys.modules['openbrowser.utils_module'] = _parent_utils
        _spec.loader.exec_module(_parent_utils)

# Re-export commonly used utilities
if _parent_utils:
    _log_pretty_path = _parent_utils._log_pretty_path
    _log_pretty_url = _parent_utils._log_pretty_url
    logger = _parent_utils.logger
    time_execution_sync = _parent_utils.time_execution_sync
    time_execution_async = _parent_utils.time_execution_async
    get_openbrowser_version = _parent_utils.get_openbrowser_version
    match_url_with_domain_pattern = _parent_utils.match_url_with_domain_pattern
    is_new_tab_page = _parent_utils.is_new_tab_page
    singleton = _parent_utils.singleton
    check_env_variables = _parent_utils.check_env_variables
    merge_dicts = _parent_utils.merge_dicts
    check_latest_openbrowser_version = _parent_utils.check_latest_openbrowser_version
    get_git_info = _parent_utils.get_git_info
    is_unsafe_pattern = _parent_utils.is_unsafe_pattern
    URL_PATTERN = _parent_utils.URL_PATTERN
    _IS_WINDOWS = _parent_utils._IS_WINDOWS
else:
    # Fallback logger if utils.py doesn't exist
    logger = logging.getLogger('openbrowser')
    _log_pretty_path = lambda x: str(x) if x else ''
    _log_pretty_url = lambda s, max_len=22: s[:max_len] + '...' if len(s) > max_len else s
    time_execution_sync = lambda x='': lambda f: f
    time_execution_async = lambda x='': lambda f: f
    get_openbrowser_version = lambda: 'unknown'
    match_url_with_domain_pattern = lambda url, pattern, log_warnings=False: False
    is_new_tab_page = lambda url: url in ('about:blank', 'chrome://new-tab-page/', 'chrome://newtab/')
    singleton = lambda cls: cls
    check_env_variables = lambda keys, any_or_all=all: False
    merge_dicts = lambda a, b, path=(): a
    check_latest_openbrowser_version = lambda: None
    get_git_info = lambda: None
    is_unsafe_pattern = lambda pattern: False
    URL_PATTERN = None
    _IS_WINDOWS = False  # Fallback for Windows check

__all__ = [
    "SignalHandler",
    "AsyncSignalHandler",
    "_log_pretty_path",
    "_log_pretty_url",
    "logger",
    "time_execution_sync",
    "time_execution_async",
    "get_openbrowser_version",
    "match_url_with_domain_pattern",
    "is_new_tab_page",
    "singleton",
    "check_env_variables",
    "merge_dicts",
    "check_latest_openbrowser_version",
    "get_git_info",
    "is_unsafe_pattern",
    "URL_PATTERN",
    "_IS_WINDOWS",
]

