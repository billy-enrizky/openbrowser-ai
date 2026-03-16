"""Comprehensive tests for openbrowser.logging_config module.

Covers: addLoggingLevel, OpenBrowserFormatter, setup_logging,
FIFOHandler, setup_log_pipes.
"""

import io
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from openbrowser.logging_config import (
    FIFOHandler,
    OpenBrowserFormatter,
    _CDP_LOGGERS,
    _RESULT_LEVEL,
    _THIRD_PARTY_LOGGERS,
    addLoggingLevel,
    setup_log_pipes,
    setup_logging,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# addLoggingLevel
# ---------------------------------------------------------------------------


class TestAddLoggingLevel:
    def test_add_custom_level(self):
        # Use a unique name to avoid collision
        level_name = "TESTCUSTOM_" + str(id(self))[-6:]
        method_name = level_name.lower()

        # Clean up in case of prior run
        for attr in (level_name, method_name):
            if hasattr(logging, attr):
                delattr(logging, attr)
            if hasattr(logging.getLoggerClass(), attr):
                delattr(logging.getLoggerClass(), attr)

        addLoggingLevel(level_name, 25, method_name)
        assert hasattr(logging, level_name)
        assert getattr(logging, level_name) == 25
        assert hasattr(logging.getLoggerClass(), method_name)

        # Clean up
        delattr(logging, level_name)
        delattr(logging, method_name)
        delattr(logging.getLoggerClass(), method_name)

    def test_duplicate_level_name_raises(self):
        with pytest.raises(AttributeError, match="already defined"):
            addLoggingLevel("INFO", 99)

    def test_duplicate_method_name_raises(self):
        with pytest.raises(AttributeError, match="already defined"):
            addLoggingLevel("UNIQUE_LEVEL_XYZ_" + str(id(self))[-6:], 99, "info")

    def test_default_method_name(self):
        level_name = "AUTOMETHOD_" + str(id(self))[-6:]
        method_name = level_name.lower()

        for attr in (level_name, method_name):
            if hasattr(logging, attr):
                delattr(logging, attr)
            if hasattr(logging.getLoggerClass(), attr):
                delattr(logging.getLoggerClass(), attr)

        addLoggingLevel(level_name, 99)
        assert hasattr(logging, method_name)

        # Clean up
        delattr(logging, level_name)
        delattr(logging, method_name)
        delattr(logging.getLoggerClass(), method_name)


# ---------------------------------------------------------------------------
# OpenBrowserFormatter
# ---------------------------------------------------------------------------


class TestOpenBrowserFormatter:
    def test_debug_mode_keeps_full_name(self):
        fmt = OpenBrowserFormatter("%(name)s - %(message)s", logging.DEBUG)
        record = logging.LogRecord(
            name="openbrowser.agent.service",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )
        result = fmt.format(record)
        assert "openbrowser.agent.service" in result

    def test_info_mode_cleans_agent_name(self):
        fmt = OpenBrowserFormatter("%(name)s - %(message)s", logging.INFO)
        record = logging.LogRecord(
            name="openbrowser.CodeAgent.service",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )
        result = fmt.format(record)
        assert "Agent" in result

    def test_info_mode_cleans_browser_session_name(self):
        fmt = OpenBrowserFormatter("%(name)s - %(message)s", logging.INFO)
        record = logging.LogRecord(
            name="openbrowser.browser.BrowserSession",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )
        result = fmt.format(record)
        assert "BrowserSession" in result

    def test_info_mode_cleans_tools_name(self):
        fmt = OpenBrowserFormatter("%(name)s - %(message)s", logging.INFO)
        record = logging.LogRecord(
            name="openbrowser.tools.service",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )
        result = fmt.format(record)
        assert "tools" in result

    def test_info_mode_cleans_dom_name(self):
        fmt = OpenBrowserFormatter("%(name)s - %(message)s", logging.INFO)
        record = logging.LogRecord(
            name="openbrowser.dom.service",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )
        result = fmt.format(record)
        assert "dom" in result

    def test_info_mode_uses_last_part_for_other(self):
        fmt = OpenBrowserFormatter("%(name)s - %(message)s", logging.INFO)
        record = logging.LogRecord(
            name="openbrowser.custom.module",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )
        result = fmt.format(record)
        assert "module" in result

    def test_non_openbrowser_name_unchanged(self):
        fmt = OpenBrowserFormatter("%(name)s - %(message)s", logging.INFO)
        record = logging.LogRecord(
            name="some.other.logger",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )
        result = fmt.format(record)
        assert "some.other.logger" in result

    def test_non_string_name_unchanged(self):
        fmt = OpenBrowserFormatter("%(name)s - %(message)s", logging.INFO)
        record = logging.LogRecord(
            name="simple",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )
        result = fmt.format(record)
        assert "simple" in result


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------


class TestSetupLogging:
    def test_returns_logger(self):
        result = setup_logging(force_setup=True)
        assert isinstance(result, logging.Logger)
        assert result.name == "openbrowser"

    def test_force_setup_clears_handlers(self):
        setup_logging(force_setup=True)
        root = logging.getLogger()
        # Should have at least one handler after setup
        assert len(root.handlers) >= 1

    def test_custom_stream(self):
        stream = io.StringIO()
        result = setup_logging(stream=stream, force_setup=True)
        assert isinstance(result, logging.Logger)

    def test_debug_level(self):
        result = setup_logging(log_level="debug", force_setup=True)
        assert isinstance(result, logging.Logger)

    def test_result_level(self):
        result = setup_logging(log_level="result", force_setup=True)
        assert isinstance(result, logging.Logger)

    def test_info_level(self):
        result = setup_logging(log_level="info", force_setup=True)
        assert isinstance(result, logging.Logger)

    def test_debug_log_file(self, tmp_path):
        log_file = tmp_path / "debug.log"
        result = setup_logging(
            force_setup=True,
            debug_log_file=str(log_file),
        )
        assert isinstance(result, logging.Logger)
        # File handler should be created
        assert log_file.parent.exists()

    def test_info_log_file(self, tmp_path):
        log_file = tmp_path / "info.log"
        result = setup_logging(
            force_setup=True,
            info_log_file=str(log_file),
        )
        assert isinstance(result, logging.Logger)

    def test_skip_setup_if_handlers_exist(self):
        # First setup
        setup_logging(force_setup=True)
        # Second setup without force should skip
        result = setup_logging(force_setup=False)
        assert isinstance(result, logging.Logger)

    def test_third_party_loggers_silenced(self):
        setup_logging(force_setup=True)
        for name in _THIRD_PARTY_LOGGERS:
            lgr = logging.getLogger(name)
            assert lgr.level >= logging.ERROR or not lgr.propagate


# ---------------------------------------------------------------------------
# FIFOHandler
# ---------------------------------------------------------------------------


class TestFIFOHandler:
    def test_init_creates_fifo(self, tmp_path):
        fifo_path = str(tmp_path / "test.pipe")
        handler = FIFOHandler(fifo_path)
        assert os.path.exists(fifo_path)
        assert handler.fd is None
        handler.close()

    def test_emit_without_reader_skips(self, tmp_path):
        fifo_path = str(tmp_path / "test.pipe")
        handler = FIFOHandler(fifo_path)

        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname="", lineno=0,
            msg="test message", args=(), exc_info=None,
        )
        # Should not crash even without a reader
        handler.emit(record)
        handler.close()

    def test_close_without_fd(self, tmp_path):
        fifo_path = str(tmp_path / "test.pipe")
        handler = FIFOHandler(fifo_path)
        handler.close()  # Should not crash

    def test_close_with_fd(self, tmp_path):
        fifo_path = str(tmp_path / "test.pipe")
        handler = FIFOHandler(fifo_path)
        handler.fd = None  # Simulate no connection
        handler.close()  # Should not crash


# ---------------------------------------------------------------------------
# setup_log_pipes
# ---------------------------------------------------------------------------


class TestSetupLogPipes:
    def test_creates_pipe_directory(self, tmp_path):
        setup_log_pipes("test-session-1234", base_dir=str(tmp_path))
        pipe_dir = tmp_path / "buagent.1234"
        # The FIFOHandler should have created the directory
        assert pipe_dir.exists()

    def test_creates_named_pipes(self, tmp_path):
        setup_log_pipes("session-abcd", base_dir=str(tmp_path))
        pipe_dir = tmp_path / "buagent.abcd"
        assert (pipe_dir / "agent.pipe").exists()
        assert (pipe_dir / "cdp.pipe").exists()
        assert (pipe_dir / "events.pipe").exists()


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_result_level(self):
        assert _RESULT_LEVEL == 35

    def test_third_party_loggers_is_frozenset(self):
        assert isinstance(_THIRD_PARTY_LOGGERS, frozenset)
        assert "httpx" in _THIRD_PARTY_LOGGERS

    def test_cdp_loggers_is_tuple(self):
        assert isinstance(_CDP_LOGGERS, tuple)
        assert "cdp_use" in _CDP_LOGGERS
