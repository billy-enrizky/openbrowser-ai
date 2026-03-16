"""Deep coverage tests for logging_config, default_action_watchdog, and browser session.

Targets:
- logging_config.py missing lines: 94, 100-101, 104, 164-165, 248, 309-310, 317-318
- default_action_watchdog.py: additional uncovered methods/branches
- session.py: additional uncovered methods/branches
"""

import asyncio
import logging
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, create_autospec, patch

import pytest
from bubus import EventBus

from openbrowser.browser.session import BrowserSession
from openbrowser.browser.views import BrowserError, URLNotAllowedError

logger = logging.getLogger(__name__)


# ===========================================================================
# logging_config.py deep coverage
# ===========================================================================


class TestAddLoggingLevelEdgeCases:
    """Test addLoggingLevel edge cases for lines 94, 100-101, 104."""

    def test_existing_method_name_on_logger_class_raises(self):
        """Line 94: methodName already exists on logger class."""
        from openbrowser.logging_config import addLoggingLevel

        # 'info' method already exists on Logger class
        with pytest.raises(AttributeError, match="already defined"):
            unique_name = "UNIQUE_LEVEL_NAME_XYZ123"
            # Ensure level name doesn't conflict
            if hasattr(logging, unique_name):
                delattr(logging, unique_name)
            addLoggingLevel(unique_name, 77, "info")

    def test_log_for_level_method_works(self):
        """Lines 100-101: logForLevel method on logger instance."""
        from openbrowser.logging_config import addLoggingLevel

        level_name = "TESTDEEP_" + str(id(self))[-6:]
        method_name = level_name.lower()

        # Clean up in case of prior run
        for attr in (level_name, method_name):
            if hasattr(logging, attr):
                delattr(logging, attr)
            if hasattr(logging.getLoggerClass(), attr):
                delattr(logging.getLoggerClass(), attr)

        addLoggingLevel(level_name, 25, method_name)

        test_logger = logging.getLogger("test_deep_level")
        test_logger.setLevel(25)
        handler = logging.StreamHandler()
        handler.setLevel(25)
        test_logger.addHandler(handler)

        # Call the dynamically added method
        log_method = getattr(test_logger, method_name)
        log_method("test message")  # Should not raise

        # Clean up
        delattr(logging, level_name)
        delattr(logging, method_name)
        delattr(logging.getLoggerClass(), method_name)

    def test_log_to_root_method_works(self):
        """Lines 103-104: logToRoot method on logging module."""
        from openbrowser.logging_config import addLoggingLevel

        level_name = "TESTROOTLOG_" + str(id(self))[-6:]
        method_name = level_name.lower()

        for attr in (level_name, method_name):
            if hasattr(logging, attr):
                delattr(logging, attr)
            if hasattr(logging.getLoggerClass(), attr):
                delattr(logging.getLoggerClass(), attr)

        addLoggingLevel(level_name, 26, method_name)

        # Call the root-level log method
        root_method = getattr(logging, method_name)
        root_method("root test message")  # Should not raise

        # Clean up
        delattr(logging, level_name)
        delattr(logging, method_name)
        delattr(logging.getLoggerClass(), method_name)


class TestSetupLoggingResultLevel:
    """Test setup_logging with RESULT level for line 164-165."""

    def test_result_level_already_exists(self):
        """Lines 164-165: RESULT level already added (AttributeError caught)."""
        from openbrowser.logging_config import setup_logging, _result_level_added
        import openbrowser.logging_config as lc

        # Ensure RESULT level is already added
        assert hasattr(logging, "RESULT")
        # Force re-add attempt
        orig_flag = lc._result_level_added
        lc._result_level_added = False
        try:
            result = setup_logging(force_setup=True, log_level="result")
            assert isinstance(result, logging.Logger)
            # Flag should be True after attempt
            assert lc._result_level_added is True
        finally:
            lc._result_level_added = orig_flag


class TestSetupLoggingCDPImportError:
    """Test setup_logging when cdp_use.logging is not available (line 248)."""

    def test_cdp_logging_import_error_fallback(self):
        """Line 248-259: ImportError fallback for CDP logging."""
        from openbrowser.logging_config import setup_logging

        with patch.dict(sys.modules, {"cdp_use.logging": None}):
            with patch(
                "builtins.__import__",
                side_effect=lambda name, *args, **kwargs: (
                    __builtins__.__import__(name, *args, **kwargs)
                    if name != "cdp_use.logging"
                    else (_ for _ in ()).throw(ImportError("no cdp_use.logging"))
                ),
            ):
                # This would use the fallback path for CDP logger configuration
                result = setup_logging(force_setup=True, log_level="info")
                assert isinstance(result, logging.Logger)


class TestFIFOHandlerDeepCoverage:
    """Test FIFOHandler for lines 309-310, 317-318."""

    def test_emit_broken_pipe_closes_fd(self, tmp_path):
        """Lines 304-311: BrokenPipeError during emit closes fd."""
        from openbrowser.logging_config import FIFOHandler

        fifo_path = str(tmp_path / "test.pipe")
        handler = FIFOHandler(fifo_path)
        # Simulate fd is open
        handler.fd = 999

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )

        with patch("os.write", side_effect=BrokenPipeError("broken pipe")):
            with patch("os.close") as mock_close:
                handler.emit(record)
                mock_close.assert_called_once_with(999)
                assert handler.fd is None

    def test_emit_os_error_during_close(self, tmp_path):
        """Lines 309-310: os.close raises during fd cleanup."""
        from openbrowser.logging_config import FIFOHandler

        fifo_path = str(tmp_path / "test.pipe")
        handler = FIFOHandler(fifo_path)
        handler.fd = 999

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )

        with patch("os.write", side_effect=OSError("write failed")):
            with patch("os.close", side_effect=OSError("close failed")):
                handler.emit(record)
                assert handler.fd is None

    def test_close_with_fd_os_error(self, tmp_path):
        """Lines 317-318: close() when fd is set and os.close raises."""
        from openbrowser.logging_config import FIFOHandler

        fifo_path = str(tmp_path / "test.pipe")
        handler = FIFOHandler(fifo_path)
        handler.fd = 888

        with patch("os.close", side_effect=OSError("close fail")):
            handler.close()  # Should not raise

    def test_emit_first_write_os_error(self, tmp_path):
        """Lines 296-300: first write attempt when no reader connected."""
        from openbrowser.logging_config import FIFOHandler

        fifo_path = str(tmp_path / "test.pipe")
        handler = FIFOHandler(fifo_path)
        assert handler.fd is None

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )

        with patch("os.open", side_effect=OSError("no reader")):
            handler.emit(record)
            assert handler.fd is None

    def test_emit_successful_write(self, tmp_path):
        """Lines 302-303: successful write to fd."""
        from openbrowser.logging_config import FIFOHandler

        fifo_path = str(tmp_path / "test.pipe")
        handler = FIFOHandler(fifo_path)
        handler.fd = 777

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))

        with patch("os.write") as mock_write:
            handler.emit(record)
            mock_write.assert_called_once()


class TestOpenBrowserFormatterEdgeCases:
    """Test OpenBrowserFormatter edge cases."""

    def test_format_non_string_name(self):
        """Test formatter with non-string record name."""
        from openbrowser.logging_config import OpenBrowserFormatter

        fmt = OpenBrowserFormatter("%(name)s - %(message)s", logging.INFO)
        record = logging.LogRecord(
            name=123,  # Non-string name
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )
        # Should not crash
        result = fmt.format(record)
        assert "test" in result

    def test_format_openbrowser_single_part(self):
        """Test formatter with openbrowser. prefix but only one part after."""
        from openbrowser.logging_config import OpenBrowserFormatter

        fmt = OpenBrowserFormatter("%(name)s - %(message)s", logging.INFO)
        record = logging.LogRecord(
            name="openbrowser.singlepart",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )
        result = fmt.format(record)
        assert "singlepart" in result


# ===========================================================================
# default_action_watchdog.py deep coverage
# ===========================================================================


def _make_element_node(**overrides):
    """Create a mock EnhancedDOMTreeNode."""
    node = MagicMock()
    node.backend_node_id = overrides.get("backend_node_id", 123)
    node.tag_name = overrides.get("tag_name", "button")
    node.node_name = overrides.get("node_name", "BUTTON")
    node.attributes = overrides.get("attributes", {})
    node.xpath = overrides.get("xpath", "/html/body/button")
    node.frame_id = overrides.get("frame_id", None)
    node.target_id = overrides.get("target_id", None)
    node.is_scrollable = overrides.get("is_scrollable", False)
    node.get_all_children_text = MagicMock(return_value="Click me")
    return node


def _make_mock_browser_session():
    session = create_autospec(BrowserSession, instance=True)
    session.logger = logging.getLogger("test_deep_daw")
    session.event_bus = MagicMock()
    session._cdp_client_root = MagicMock()
    session.agent_focus = MagicMock()
    session.agent_focus.target_id = "AAAA1111BBBB2222CCCC3333DDDD4444"
    session.agent_focus.session_id = "sess-deep-1234"
    session.agent_focus.cdp_client = MagicMock()
    session.browser_profile = MagicMock()
    session.browser_profile.downloads_path = "/tmp/downloads"
    session.is_file_input = MagicMock(return_value=False)
    session.cdp_client = MagicMock()
    session.cdp_client_for_node = AsyncMock()
    session.get_element_coordinates = AsyncMock()
    session.get_or_create_cdp_session = AsyncMock()
    session.get_current_page_url = AsyncMock(return_value="https://example.com")
    session.get_current_page_title = AsyncMock(return_value="Test Page")
    return session


def _make_watchdog():
    """Create DefaultActionWatchdog with mocked browser session."""
    from openbrowser.browser.watchdogs.default_action_watchdog import (
        DefaultActionWatchdog,
    )

    session = _make_mock_browser_session()
    watchdog = DefaultActionWatchdog(
        event_bus=EventBus(), browser_session=session
    )
    watchdog.event_bus = MagicMock()
    return watchdog


class TestRequiresDirectValueAssignment:
    """Test _requires_direct_value_assignment for various input types."""

    def test_date_input(self):
        wd = _make_watchdog()
        node = _make_element_node(
            tag_name="input", attributes={"type": "date"}
        )
        assert wd._requires_direct_value_assignment(node) is True

    def test_time_input(self):
        wd = _make_watchdog()
        node = _make_element_node(
            tag_name="input", attributes={"type": "time"}
        )
        assert wd._requires_direct_value_assignment(node) is True

    def test_datetime_local_input(self):
        wd = _make_watchdog()
        node = _make_element_node(
            tag_name="input", attributes={"type": "datetime-local"}
        )
        assert wd._requires_direct_value_assignment(node) is True

    def test_month_input(self):
        wd = _make_watchdog()
        node = _make_element_node(
            tag_name="input", attributes={"type": "month"}
        )
        assert wd._requires_direct_value_assignment(node) is True

    def test_week_input(self):
        wd = _make_watchdog()
        node = _make_element_node(
            tag_name="input", attributes={"type": "week"}
        )
        assert wd._requires_direct_value_assignment(node) is True

    def test_color_input(self):
        wd = _make_watchdog()
        node = _make_element_node(
            tag_name="input", attributes={"type": "color"}
        )
        assert wd._requires_direct_value_assignment(node) is True

    def test_range_input(self):
        wd = _make_watchdog()
        node = _make_element_node(
            tag_name="input", attributes={"type": "range"}
        )
        assert wd._requires_direct_value_assignment(node) is True

    def test_text_input_with_datepicker_class(self):
        wd = _make_watchdog()
        node = _make_element_node(
            tag_name="input",
            attributes={"type": "text", "class": "form-control datepicker"},
        )
        assert wd._requires_direct_value_assignment(node) is True

    def test_text_input_with_data_datepicker(self):
        wd = _make_watchdog()
        node = _make_element_node(
            tag_name="input",
            attributes={"type": "text", "data-datepicker": "true"},
        )
        assert wd._requires_direct_value_assignment(node) is True

    def test_text_input_with_data_date_format(self):
        wd = _make_watchdog()
        node = _make_element_node(
            tag_name="input",
            attributes={"type": "text", "data-date-format": "YYYY-MM-DD"},
        )
        assert wd._requires_direct_value_assignment(node) is True

    def test_text_input_with_data_provide(self):
        wd = _make_watchdog()
        node = _make_element_node(
            tag_name="input",
            attributes={"type": "text", "data-provide": "datepicker"},
        )
        assert wd._requires_direct_value_assignment(node) is True

    def test_regular_text_input(self):
        wd = _make_watchdog()
        node = _make_element_node(
            tag_name="input", attributes={"type": "text"}
        )
        assert wd._requires_direct_value_assignment(node) is False

    def test_no_tag_name(self):
        wd = _make_watchdog()
        node = _make_element_node(tag_name=None, attributes={"type": "text"})
        assert wd._requires_direct_value_assignment(node) is False

    def test_no_attributes(self):
        wd = _make_watchdog()
        node = _make_element_node(tag_name="input", attributes=None)
        assert wd._requires_direct_value_assignment(node) is False

    def test_non_input_element(self):
        wd = _make_watchdog()
        node = _make_element_node(tag_name="div", attributes={"type": "date"})
        assert wd._requires_direct_value_assignment(node) is False

    def test_empty_type_with_datepicker_class(self):
        wd = _make_watchdog()
        node = _make_element_node(
            tag_name="input",
            attributes={"type": "", "class": "datetimepicker"},
        )
        assert wd._requires_direct_value_assignment(node) is True


class TestGetCharModifiersAndVk:
    """Test _get_char_modifiers_and_vk method."""

    def test_shift_char_bang(self):
        wd = _make_watchdog()
        modifiers, vk, base = wd._get_char_modifiers_and_vk("!")
        assert modifiers == 8  # Shift
        assert base == "1"

    def test_at_sign(self):
        wd = _make_watchdog()
        modifiers, vk, base = wd._get_char_modifiers_and_vk("@")
        assert modifiers == 8
        assert base == "2"

    def test_uppercase_letter(self):
        wd = _make_watchdog()
        modifiers, vk, base = wd._get_char_modifiers_and_vk("A")
        assert modifiers == 8
        assert base == "a"

    def test_lowercase_letter(self):
        wd = _make_watchdog()
        modifiers, vk, base = wd._get_char_modifiers_and_vk("a")
        assert modifiers == 0
        assert vk == ord("A")

    def test_digit(self):
        wd = _make_watchdog()
        modifiers, vk, base = wd._get_char_modifiers_and_vk("5")
        assert modifiers == 0
        assert base == "5"

    def test_space(self):
        wd = _make_watchdog()
        modifiers, vk, base = wd._get_char_modifiers_and_vk(" ")
        assert modifiers == 0
        assert vk == 32

    def test_semicolon(self):
        wd = _make_watchdog()
        modifiers, vk, base = wd._get_char_modifiers_and_vk(";")
        assert modifiers == 0

    def test_tilde(self):
        wd = _make_watchdog()
        modifiers, vk, base = wd._get_char_modifiers_and_vk("~")
        assert modifiers == 8

    def test_all_shift_chars(self):
        wd = _make_watchdog()
        shift_chars = "!@#$%^&*()_+{}|:\"<>?~"
        for ch in shift_chars:
            modifiers, _, _ = wd._get_char_modifiers_and_vk(ch)
            assert modifiers == 8, f"Expected Shift for '{ch}'"


class TestGetKeyCodeForChar:
    """Test _get_key_code_for_char method."""

    def test_space(self):
        wd = _make_watchdog()
        assert wd._get_key_code_for_char(" ") == "Space"

    def test_period(self):
        wd = _make_watchdog()
        assert wd._get_key_code_for_char(".") == "Period"

    def test_digit(self):
        wd = _make_watchdog()
        assert wd._get_key_code_for_char("5") == "Digit5"

    def test_letter(self):
        wd = _make_watchdog()
        assert wd._get_key_code_for_char("a") == "KeyA"

    def test_uppercase_letter(self):
        wd = _make_watchdog()
        assert wd._get_key_code_for_char("A") == "KeyA"

    def test_at_sign(self):
        wd = _make_watchdog()
        assert wd._get_key_code_for_char("@") == "Digit2"

    def test_slash(self):
        wd = _make_watchdog()
        assert wd._get_key_code_for_char("/") == "Slash"

    def test_unknown_char_fallback(self):
        wd = _make_watchdog()
        # Non-mapped character
        result = wd._get_key_code_for_char("\x01")
        assert result.startswith("Key")


class TestOnScrollEvent:
    """Test on_ScrollEvent method."""

    @pytest.mark.asyncio
    async def test_scroll_no_agent_focus(self):
        """Line 280-282: no agent focus raises BrowserError."""
        wd = _make_watchdog()
        wd.browser_session.agent_focus = None

        from openbrowser.browser.events import ScrollEvent

        event = ScrollEvent(direction="down", amount=300, node=None)

        with pytest.raises(BrowserError, match="No active target"):
            await wd.on_ScrollEvent(event)

    @pytest.mark.asyncio
    async def test_scroll_down_no_node(self):
        """Lines 316-324: scroll down without specific node."""
        wd = _make_watchdog()

        from openbrowser.browser.events import ScrollEvent

        event = ScrollEvent(direction="down", amount=300, node=None)

        with patch.object(
            wd, "_scroll_with_cdp_gesture", new_callable=AsyncMock, return_value=True
        ):
            result = await wd.on_ScrollEvent(event)
            assert result is None

    @pytest.mark.asyncio
    async def test_scroll_up_no_node(self):
        """Line 287: scroll up (negative pixels)."""
        wd = _make_watchdog()

        from openbrowser.browser.events import ScrollEvent

        event = ScrollEvent(direction="up", amount=200, node=None)

        with patch.object(
            wd, "_scroll_with_cdp_gesture", new_callable=AsyncMock, return_value=True
        ):
            result = await wd.on_ScrollEvent(event)
            assert result is None

    @pytest.mark.asyncio
    async def test_scroll_with_element_node(self):
        """Lines 290-314: scroll specific element."""
        wd = _make_watchdog()

        node = _make_element_node(tag_name="div")

        from openbrowser.browser.events import ScrollEvent

        event = ScrollEvent.model_construct(direction="down", amount=300, node=node)

        with patch.object(
            wd,
            "_scroll_element_container",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await wd.on_ScrollEvent(event)
            assert result is None

    @pytest.mark.asyncio
    async def test_scroll_iframe_forces_dom_refresh(self):
        """Lines 295-313: scrolling iframe element."""
        wd = _make_watchdog()

        node = _make_element_node(tag_name="IFRAME")

        from openbrowser.browser.events import ScrollEvent

        event = ScrollEvent.model_construct(direction="down", amount=300, node=node)

        with patch.object(
            wd,
            "_scroll_element_container",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await wd.on_ScrollEvent(event)
            assert result is None

    @pytest.mark.asyncio
    async def test_scroll_element_container_fails_falls_back(self):
        """Lines 316-317: element scroll fails, falls back to page scroll."""
        wd = _make_watchdog()

        node = _make_element_node(tag_name="div")

        from openbrowser.browser.events import ScrollEvent

        event = ScrollEvent.model_construct(direction="down", amount=300, node=node)

        with patch.object(
            wd,
            "_scroll_element_container",
            new_callable=AsyncMock,
            return_value=False,
        ):
            with patch.object(
                wd,
                "_scroll_with_cdp_gesture",
                new_callable=AsyncMock,
                return_value=True,
            ):
                result = await wd.on_ScrollEvent(event)
                assert result is None


class TestTypeToPage:
    """Test _type_to_page method."""

    @pytest.mark.asyncio
    async def test_type_regular_chars(self):
        """Lines 728-786: type regular characters to page."""
        wd = _make_watchdog()

        mock_cdp_session = MagicMock()
        mock_cdp_session.cdp_client.send.Input.dispatchKeyEvent = AsyncMock()
        mock_cdp_session.session_id = "sess-1"
        wd.browser_session.get_or_create_cdp_session = AsyncMock(
            return_value=mock_cdp_session
        )

        await wd._type_to_page("ab")
        # 3 events per char (keyDown, char, keyUp) * 2 chars = 6
        assert mock_cdp_session.cdp_client.send.Input.dispatchKeyEvent.call_count == 6

    @pytest.mark.asyncio
    async def test_type_newline(self):
        """Lines 730-758: type newline to page."""
        wd = _make_watchdog()

        mock_cdp_session = MagicMock()
        mock_cdp_session.cdp_client.send.Input.dispatchKeyEvent = AsyncMock()
        mock_cdp_session.session_id = "sess-1"
        wd.browser_session.get_or_create_cdp_session = AsyncMock(
            return_value=mock_cdp_session
        )

        await wd._type_to_page("\n")
        # 3 events for newline (keyDown, char, keyUp)
        assert mock_cdp_session.cdp_client.send.Input.dispatchKeyEvent.call_count == 3

    @pytest.mark.asyncio
    async def test_type_to_page_exception(self):
        """Lines 788-789: exception during typing."""
        wd = _make_watchdog()

        wd.browser_session.get_or_create_cdp_session = AsyncMock(
            side_effect=Exception("no session")
        )

        with pytest.raises(Exception, match="Failed to type to page"):
            await wd._type_to_page("abc")


class TestSetValueDirectly:
    """Test _set_value_directly method."""

    @pytest.mark.asyncio
    async def test_set_value_success(self):
        """Lines 1233-1299: successful value setting."""
        wd = _make_watchdog()
        node = _make_element_node(tag_name="input", attributes={"type": "date"})

        mock_cdp = MagicMock()
        mock_cdp.cdp_client.send.Runtime.callFunctionOn = AsyncMock(
            return_value={
                "result": {"value": "2025-01-01"}
            }
        )
        mock_cdp.session_id = "sess-1"

        await wd._set_value_directly(node, "2025-01-01", "obj-1", mock_cdp)

    @pytest.mark.asyncio
    async def test_set_value_no_verification(self):
        """Lines 1300-1301: value set but cannot verify."""
        wd = _make_watchdog()
        node = _make_element_node(tag_name="input", attributes={"type": "date"})

        mock_cdp = MagicMock()
        mock_cdp.cdp_client.send.Runtime.callFunctionOn = AsyncMock(
            return_value={}
        )
        mock_cdp.session_id = "sess-1"

        await wd._set_value_directly(node, "2025-01-01", "obj-1", mock_cdp)

    @pytest.mark.asyncio
    async def test_set_value_exception(self):
        """Lines 1303-1305: exception during value setting."""
        wd = _make_watchdog()
        node = _make_element_node(tag_name="input", attributes={"type": "date"})

        mock_cdp = MagicMock()
        mock_cdp.cdp_client.send.Runtime.callFunctionOn = AsyncMock(
            side_effect=RuntimeError("js failed")
        )
        mock_cdp.session_id = "sess-1"

        with pytest.raises(RuntimeError, match="js failed"):
            await wd._set_value_directly(node, "2025-01-01", "obj-1", mock_cdp)


class TestTriggerFrameworkEvents:
    """Test _trigger_framework_events method."""

    @pytest.mark.asyncio
    async def test_trigger_success(self):
        """Lines 1523-1616: successful framework event trigger."""
        wd = _make_watchdog()

        mock_cdp = MagicMock()
        mock_cdp.cdp_client.send.Runtime.callFunctionOn = AsyncMock(
            return_value={"result": {"value": True}}
        )
        mock_cdp.session_id = "sess-1"

        await wd._trigger_framework_events("obj-1", mock_cdp)

    @pytest.mark.asyncio
    async def test_trigger_exception_swallowed(self):
        """Lines 1618-1620: exception swallowed during framework events."""
        wd = _make_watchdog()

        mock_cdp = MagicMock()
        mock_cdp.cdp_client.send.Runtime.callFunctionOn = AsyncMock(
            side_effect=Exception("js error")
        )
        mock_cdp.session_id = "sess-1"

        # Should not raise
        await wd._trigger_framework_events("obj-1", mock_cdp)


class TestScrollWithCDPGesture:
    """Test _scroll_with_cdp_gesture method."""

    @pytest.mark.asyncio
    async def test_scroll_success(self):
        """Lines 1632-1663: successful scroll."""
        wd = _make_watchdog()
        mock_cdp = MagicMock()
        mock_cdp.send.Page.getLayoutMetrics = AsyncMock(
            return_value={
                "layoutViewport": {"clientWidth": 1920, "clientHeight": 1080}
            }
        )
        mock_cdp.send.Input.dispatchMouseEvent = AsyncMock()
        wd.browser_session.agent_focus.cdp_client = mock_cdp
        wd.browser_session.agent_focus.session_id = "sess-1"

        result = await wd._scroll_with_cdp_gesture(300)
        assert result is True

    @pytest.mark.asyncio
    async def test_scroll_exception(self):
        """Lines 1665-1667: scroll fails."""
        wd = _make_watchdog()
        wd.browser_session.agent_focus.cdp_client.send.Page.getLayoutMetrics = (
            AsyncMock(side_effect=Exception("error"))
        )

        result = await wd._scroll_with_cdp_gesture(300)
        assert result is False

    @pytest.mark.asyncio
    async def test_scroll_no_agent_focus(self):
        """Line 1634: no agent_focus -- assertion caught by except, returns False."""
        wd = _make_watchdog()
        wd.browser_session.agent_focus = None

        result = await wd._scroll_with_cdp_gesture(300)
        assert result is False


class TestCheckElementOcclusion:
    """Test _check_element_occlusion method."""

    @pytest.mark.asyncio
    async def test_no_object_in_resolve(self):
        """Lines 350-352: resolveNode returns no 'object'."""
        wd = _make_watchdog()
        mock_cdp = MagicMock()
        mock_cdp.cdp_client.send.DOM.resolveNode = AsyncMock(return_value={})
        mock_cdp.session_id = "sess-1"

        result = await wd._check_element_occlusion(123, 100, 100, mock_cdp)
        assert result is True

    @pytest.mark.asyncio
    async def test_element_is_clickable(self):
        """Lines 401-405: element is clickable (not occluded)."""
        wd = _make_watchdog()
        mock_cdp = MagicMock()
        mock_cdp.cdp_client.send.DOM.resolveNode = AsyncMock(
            return_value={"object": {"objectId": "obj-1"}}
        )
        mock_cdp.cdp_client.send.Runtime.callFunctionOn = AsyncMock(
            return_value={
                "result": {
                    "value": {
                        "isClickable": True,
                        "targetInfo": {"tagName": "BUTTON"},
                    }
                }
            }
        )
        mock_cdp.session_id = "sess-1"

        result = await wd._check_element_occlusion(123, 100, 100, mock_cdp)
        assert result is False

    @pytest.mark.asyncio
    async def test_element_is_occluded(self):
        """Lines 406-415: element is occluded."""
        wd = _make_watchdog()
        mock_cdp = MagicMock()
        mock_cdp.cdp_client.send.DOM.resolveNode = AsyncMock(
            return_value={"object": {"objectId": "obj-1"}}
        )
        mock_cdp.cdp_client.send.Runtime.callFunctionOn = AsyncMock(
            return_value={
                "result": {
                    "value": {
                        "isClickable": False,
                        "targetInfo": {"tagName": "BUTTON", "id": "btn1"},
                        "elementAtPointInfo": {"tagName": "DIV", "id": "overlay"},
                    }
                }
            }
        )
        mock_cdp.session_id = "sess-1"

        result = await wd._check_element_occlusion(123, 100, 100, mock_cdp)
        assert result is True

    @pytest.mark.asyncio
    async def test_occlusion_check_exception(self):
        """Lines 417-419: exception during occlusion check."""
        wd = _make_watchdog()
        mock_cdp = MagicMock()
        mock_cdp.cdp_client.send.DOM.resolveNode = AsyncMock(
            side_effect=Exception("error")
        )
        mock_cdp.session_id = "sess-1"

        result = await wd._check_element_occlusion(123, 100, 100, mock_cdp)
        assert result is False

    @pytest.mark.asyncio
    async def test_no_result_in_call_function(self):
        """Lines 396-398: callFunctionOn returns no result."""
        wd = _make_watchdog()
        mock_cdp = MagicMock()
        mock_cdp.cdp_client.send.DOM.resolveNode = AsyncMock(
            return_value={"object": {"objectId": "obj-1"}}
        )
        mock_cdp.cdp_client.send.Runtime.callFunctionOn = AsyncMock(
            return_value={}
        )
        mock_cdp.session_id = "sess-1"

        result = await wd._check_element_occlusion(123, 100, 100, mock_cdp)
        assert result is True


class TestClickElementNodeImplEdges:
    """Test _click_element_node_impl edge cases."""

    @pytest.mark.asyncio
    async def test_select_element_returns_validation_error(self):
        """Lines 434-437: <select> element returns validation error."""
        wd = _make_watchdog()
        node = _make_element_node(tag_name="select", attributes={})

        result = await wd._click_element_node_impl(node)
        assert result is not None
        assert "validation_error" in result

    @pytest.mark.asyncio
    async def test_file_input_returns_validation_error(self):
        """Lines 439-442: file input returns validation error."""
        wd = _make_watchdog()
        node = _make_element_node(
            tag_name="input", attributes={"type": "file"}
        )

        result = await wd._click_element_node_impl(node)
        assert result is not None
        assert "validation_error" in result


class TestOnTypeTextEventEdges:
    """Test on_TypeTextEvent edge cases."""

    @pytest.mark.asyncio
    async def test_type_to_page_when_backend_node_is_zero(self):
        """Lines 223-234: backend_node_id is 0, type to page."""
        wd = _make_watchdog()
        node = _make_element_node(backend_node_id=0)

        from openbrowser.browser.events import TypeTextEvent

        event = TypeTextEvent.model_construct(
            text="hello",
            node=node,
            clear=False,
            is_sensitive=False,
        )

        with patch.object(wd, "_type_to_page", new_callable=AsyncMock):
            result = await wd.on_TypeTextEvent(event)
            assert result is None

    @pytest.mark.asyncio
    async def test_type_sensitive_with_key_name_to_page(self):
        """Lines 228-229: sensitive text with key name typed to page."""
        wd = _make_watchdog()
        node = _make_element_node(backend_node_id=0)

        from openbrowser.browser.events import TypeTextEvent

        event = TypeTextEvent.model_construct(
            text="secret",
            node=node,
            clear=False,
            is_sensitive=True,
            sensitive_key_name="password",
        )

        with patch.object(wd, "_type_to_page", new_callable=AsyncMock):
            result = await wd.on_TypeTextEvent(event)
            assert result is None

    @pytest.mark.asyncio
    async def test_type_sensitive_no_key_name_to_page(self):
        """Lines 230-231: sensitive text without key name typed to page."""
        wd = _make_watchdog()
        node = _make_element_node(backend_node_id=0)

        from openbrowser.browser.events import TypeTextEvent

        event = TypeTextEvent.model_construct(
            text="secret",
            node=node,
            clear=False,
            is_sensitive=True,
        )

        with patch.object(wd, "_type_to_page", new_callable=AsyncMock):
            result = await wd.on_TypeTextEvent(event)
            assert result is None

    @pytest.mark.asyncio
    async def test_type_element_fails_falls_back_to_page(self):
        """Lines 254-270: typing to element fails, falls back to page."""
        wd = _make_watchdog()
        node = _make_element_node(backend_node_id=42)

        from openbrowser.browser.events import TypeTextEvent

        event = TypeTextEvent.model_construct(
            text="hello",
            node=node,
            clear=False,
            is_sensitive=False,
        )

        with patch.object(
            wd,
            "_input_text_element_node_impl",
            new_callable=AsyncMock,
            side_effect=Exception("element not found"),
        ):
            with patch.object(
                wd,
                "_click_element_node_impl",
                new_callable=AsyncMock,
            ):
                with patch.object(
                    wd, "_type_to_page", new_callable=AsyncMock
                ):
                    result = await wd.on_TypeTextEvent(event)
                    assert result is None

    @pytest.mark.asyncio
    async def test_type_sensitive_with_key_name_to_element(self):
        """Lines 246-247: sensitive with key name to specific element."""
        wd = _make_watchdog()
        node = _make_element_node(backend_node_id=42)

        from openbrowser.browser.events import TypeTextEvent

        event = TypeTextEvent.model_construct(
            text="secret",
            node=node,
            clear=False,
            is_sensitive=True,
            sensitive_key_name="api_key",
        )

        with patch.object(
            wd,
            "_input_text_element_node_impl",
            new_callable=AsyncMock,
            return_value={"input_x": 100, "input_y": 200},
        ):
            result = await wd.on_TypeTextEvent(event)
            assert result is not None

    @pytest.mark.asyncio
    async def test_type_sensitive_no_key_name_to_element(self):
        """Lines 248-249: sensitive without key name to specific element."""
        wd = _make_watchdog()
        node = _make_element_node(backend_node_id=42)

        from openbrowser.browser.events import TypeTextEvent

        event = TypeTextEvent.model_construct(
            text="secret",
            node=node,
            clear=False,
            is_sensitive=True,
        )

        with patch.object(
            wd,
            "_input_text_element_node_impl",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await wd.on_TypeTextEvent(event)

    @pytest.mark.asyncio
    async def test_type_fallback_sensitive_with_key_name(self):
        """Lines 264-265: sensitive with key name in fallback path."""
        wd = _make_watchdog()
        node = _make_element_node(backend_node_id=42)

        from openbrowser.browser.events import TypeTextEvent

        event = TypeTextEvent.model_construct(
            text="secret",
            node=node,
            clear=False,
            is_sensitive=True,
            sensitive_key_name="password",
        )

        with patch.object(
            wd,
            "_input_text_element_node_impl",
            new_callable=AsyncMock,
            side_effect=Exception("fail"),
        ):
            with patch.object(
                wd,
                "_click_element_node_impl",
                new_callable=AsyncMock,
            ):
                with patch.object(
                    wd, "_type_to_page", new_callable=AsyncMock
                ):
                    result = await wd.on_TypeTextEvent(event)
                    assert result is None

    @pytest.mark.asyncio
    async def test_type_fallback_sensitive_no_key_name(self):
        """Lines 266-267: sensitive without key name in fallback path."""
        wd = _make_watchdog()
        node = _make_element_node(backend_node_id=42)

        from openbrowser.browser.events import TypeTextEvent

        event = TypeTextEvent.model_construct(
            text="secret",
            node=node,
            clear=False,
            is_sensitive=True,
            sensitive_key_name=None,
        )

        with patch.object(
            wd,
            "_input_text_element_node_impl",
            new_callable=AsyncMock,
            side_effect=Exception("fail"),
        ):
            with patch.object(
                wd,
                "_click_element_node_impl",
                new_callable=AsyncMock,
            ):
                with patch.object(
                    wd, "_type_to_page", new_callable=AsyncMock
                ):
                    result = await wd.on_TypeTextEvent(event)
                    assert result is None


class TestFocusElementSimple:
    """Test _focus_element_simple with different scenarios."""

    @pytest.mark.asyncio
    async def test_cdp_focus_fails_no_coordinates(self):
        """Lines 1125-1166: CDP focus fails, no coordinates provided."""
        wd = _make_watchdog()
        mock_cdp = MagicMock()
        mock_cdp.cdp_client.send.DOM.focus = AsyncMock(
            side_effect=Exception("focus failed")
        )
        mock_cdp.session_id = "sess-1"

        result = await wd._focus_element_simple(123, "obj-1", mock_cdp, None)
        assert result is False

    @pytest.mark.asyncio
    async def test_cdp_focus_fails_click_focus_succeeds(self):
        """Lines 1129-1159: CDP focus fails, click focus succeeds."""
        wd = _make_watchdog()
        mock_cdp = MagicMock()
        mock_cdp.cdp_client.send.DOM.focus = AsyncMock(
            side_effect=Exception("focus failed")
        )
        mock_cdp.cdp_client.send.Input.dispatchMouseEvent = AsyncMock()
        mock_cdp.session_id = "sess-1"

        result = await wd._focus_element_simple(
            123, "obj-1", mock_cdp, {"input_x": 100.0, "input_y": 200.0}
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_cdp_focus_fails_click_focus_also_fails(self):
        """Lines 1161-1162: both CDP and click focus fail."""
        wd = _make_watchdog()
        mock_cdp = MagicMock()
        mock_cdp.cdp_client.send.DOM.focus = AsyncMock(
            side_effect=Exception("focus failed")
        )
        mock_cdp.cdp_client.send.Input.dispatchMouseEvent = AsyncMock(
            side_effect=Exception("click failed")
        )
        mock_cdp.session_id = "sess-1"

        result = await wd._focus_element_simple(
            123, "obj-1", mock_cdp, {"input_x": 100.0, "input_y": 200.0}
        )
        assert result is False


class TestGetSessionIdForElement:
    """Test _get_session_id_for_element method."""

    @pytest.mark.asyncio
    async def test_main_frame_element(self):
        """Lines 1780-1782: element in main frame."""
        wd = _make_watchdog()

        node = _make_element_node(frame_id=None)
        result = await wd._get_session_id_for_element(node)
        assert result == wd.browser_session.agent_focus.session_id

    @pytest.mark.asyncio
    async def test_iframe_element_found(self):
        """Lines 1761-1773: element in iframe, target found."""
        wd = _make_watchdog()

        node = _make_element_node(frame_id="frame-123")
        mock_targets = {
            "targetInfos": [
                {
                    "type": "iframe",
                    "targetId": "target-frame-123-xyz",
                }
            ]
        }
        wd.browser_session.cdp_client.send.Target.getTargets = AsyncMock(
            return_value=mock_targets
        )
        mock_session = MagicMock()
        mock_session.session_id = "iframe-session-1"
        wd.browser_session.get_or_create_cdp_session = AsyncMock(
            return_value=mock_session
        )

        result = await wd._get_session_id_for_element(node)
        assert result == "iframe-session-1"

    @pytest.mark.asyncio
    async def test_iframe_element_not_found(self):
        """Lines 1775-1776: iframe frame_id not found in targets."""
        wd = _make_watchdog()

        node = _make_element_node(frame_id="frame-999")
        mock_targets = {
            "targetInfos": [
                {
                    "type": "page",
                    "targetId": "target-page-1",
                }
            ]
        }
        wd.browser_session.cdp_client.send.Target.getTargets = AsyncMock(
            return_value=mock_targets
        )

        result = await wd._get_session_id_for_element(node)
        assert result == wd.browser_session.agent_focus.session_id

    @pytest.mark.asyncio
    async def test_iframe_element_exception(self):
        """Lines 1777-1778: exception getting frame session."""
        wd = _make_watchdog()

        node = _make_element_node(frame_id="frame-123")
        wd.browser_session.cdp_client.send.Target.getTargets = AsyncMock(
            side_effect=Exception("network error")
        )

        result = await wd._get_session_id_for_element(node)
        assert result == wd.browser_session.agent_focus.session_id


# ===========================================================================
# BrowserSession deep coverage
# ===========================================================================


class TestBrowserSessionProperties:
    """Test BrowserSession simple properties."""

    def test_cdp_url_property(self):
        """Test cdp_url property delegates to browser_profile."""
        session = BrowserSession()
        assert session.cdp_url == session.browser_profile.cdp_url

    def test_is_local_property(self):
        """Test is_local property delegates to browser_profile."""
        session = BrowserSession()
        assert session.is_local == session.browser_profile.is_local

    def test_current_target_id_no_focus(self):
        """Test current_target_id returns None when no agent_focus."""
        session = BrowserSession()
        assert session.current_target_id is None

    def test_current_session_id_no_focus(self):
        """Test current_session_id returns None when no agent_focus."""
        session = BrowserSession()
        assert session.current_session_id is None

    def test_cdp_client_property_not_initialized(self):
        """Test cdp_client property raises when not initialized."""
        session = BrowserSession()
        with pytest.raises(AssertionError, match="CDP client not initialized"):
            _ = session.cdp_client

    def test_repr_and_str(self):
        """Test __repr__ and __str__."""
        session = BrowserSession()
        repr_str = repr(session)
        str_str = str(session)
        assert "BrowserSession" in repr_str
        assert "BrowserSession" in str_str


class TestBrowserSessionInit:
    """Test BrowserSession initialization edge cases."""

    def test_init_with_cdp_url(self):
        """Test init with cdp_url provided."""
        session = BrowserSession(cdp_url="ws://localhost:9222")
        assert session.cdp_url == "ws://localhost:9222"

    def test_init_with_executable_path_sets_local(self):
        """Test is_local=True when executable_path is provided."""
        session = BrowserSession(executable_path="/usr/bin/chromium")
        assert session.is_local is True

    def test_init_with_browser_profile(self):
        """Test init merging browser_profile with kwargs."""
        from openbrowser.browser.profile import BrowserProfile

        profile = BrowserProfile(headless=True)
        session = BrowserSession(browser_profile=profile, keep_alive=True)
        assert session.browser_profile.keep_alive is True

    def test_init_no_cdp_url_sets_local(self):
        """Test is_local=True when no cdp_url."""
        session = BrowserSession()
        assert session.is_local is True


class TestBrowserSessionReset:
    """Test BrowserSession reset."""

    @pytest.mark.asyncio
    async def test_reset_clears_all_state(self):
        """Test reset clears all cached state."""
        session = BrowserSession()
        session._cdp_session_pool["test"] = MagicMock()
        session._cached_browser_state_summary = MagicMock()
        session._cached_selector_map[1] = MagicMock()
        session._downloaded_files.append("file.pdf")

        await session.reset()

        assert session._cdp_client_root is None
        assert session._cached_browser_state_summary is None
        assert len(session._cached_selector_map) == 0
        assert len(session._downloaded_files) == 0
        assert session.agent_focus is None

    @pytest.mark.asyncio
    async def test_reset_with_session_manager(self):
        """Test reset clears session manager."""
        session = BrowserSession()
        mock_manager = MagicMock()
        mock_manager.clear = AsyncMock()
        session._session_manager = mock_manager

        await session.reset()

        mock_manager.clear.assert_called_once()
        assert session._session_manager is None


class TestBrowserSessionOnFileDownloaded:
    """Test on_FileDownloadedEvent handler."""

    @pytest.mark.asyncio
    async def test_track_new_download(self):
        """Lines 872-877: track new downloaded file."""
        session = BrowserSession()

        from openbrowser.browser.events import FileDownloadedEvent

        event = FileDownloadedEvent(
            url="https://example.com/file.pdf",
            path="/tmp/file.pdf",
            file_name="file.pdf",
            file_size=1024,
        )

        await session.on_FileDownloadedEvent(event)
        assert "/tmp/file.pdf" in session._downloaded_files

    @pytest.mark.asyncio
    async def test_skip_duplicate_download(self):
        """Lines 881-882: skip already tracked file."""
        session = BrowserSession()
        session._downloaded_files.append("/tmp/file.pdf")

        from openbrowser.browser.events import FileDownloadedEvent

        event = FileDownloadedEvent(
            url="https://example.com/file.pdf",
            path="/tmp/file.pdf",
            file_name="file.pdf",
            file_size=1024,
        )

        await session.on_FileDownloadedEvent(event)
        # Should not add duplicate
        assert session._downloaded_files.count("/tmp/file.pdf") == 1

    @pytest.mark.asyncio
    async def test_download_no_path(self):
        """Lines 879-880: FileDownloadedEvent with no path -- bypass validation."""
        session = BrowserSession()

        from openbrowser.browser.events import FileDownloadedEvent

        # Use model_construct to bypass validation since path is required str
        event = FileDownloadedEvent.model_construct(
            url="https://example.com/file.pdf",
            path=None,
            file_name="file.pdf",
            file_size=1024,
        )

        await session.on_FileDownloadedEvent(event)
        assert len(session._downloaded_files) == 0


class TestBrowserSessionOnBrowserStopEvent:
    """Test on_BrowserStopEvent handler."""

    @pytest.mark.asyncio
    async def test_keep_alive_not_forced(self):
        """Lines 889-891: keep_alive=True and not forced returns early."""
        session = BrowserSession(keep_alive=True)
        # Use object.__setattr__ to bypass Pydantic validate_assignment
        mock_bus = MagicMock()
        object.__setattr__(session, "event_bus", mock_bus)

        from openbrowser.browser.events import BrowserStopEvent

        event = BrowserStopEvent(force=False)
        await session.on_BrowserStopEvent(event)
        # Session should dispatch BrowserStoppedEvent
        mock_bus.dispatch.assert_called()

    @pytest.mark.asyncio
    async def test_stop_error_dispatches_error_event(self):
        """Lines 905-912: exception dispatches BrowserErrorEvent."""
        session = BrowserSession()
        mock_bus = MagicMock()
        object.__setattr__(session, "event_bus", mock_bus)

        # Bypass Pydantic validate_assignment by patching at class level
        original_reset = BrowserSession.reset
        try:
            BrowserSession.reset = AsyncMock(side_effect=Exception("reset fail"))

            from openbrowser.browser.events import BrowserStopEvent

            event = BrowserStopEvent(force=True)
            await session.on_BrowserStopEvent(event)
            # BrowserErrorEvent should have been dispatched
            assert mock_bus.dispatch.called
        finally:
            BrowserSession.reset = original_reset


class TestBrowserSessionOnTabClosedEvent:
    """Test on_TabClosedEvent handler."""

    @pytest.mark.asyncio
    async def test_no_agent_focus(self):
        """Lines 801-802: no agent_focus returns early."""
        session = BrowserSession()
        session.agent_focus = None

        from openbrowser.browser.events import TabClosedEvent

        event = TabClosedEvent(target_id="target-1")
        await session.on_TabClosedEvent(event)
        # No error

    @pytest.mark.asyncio
    async def test_current_tab_closed_switches(self):
        """Lines 808-809: closed tab was current, triggers switch."""
        session = BrowserSession()
        mock_bus = MagicMock()
        mock_bus.dispatch = AsyncMock()
        object.__setattr__(session, "event_bus", mock_bus)
        # Use object.__setattr__ to bypass Pydantic validation for agent_focus
        mock_focus = MagicMock()
        mock_focus.target_id = "target-1"
        object.__setattr__(session, "agent_focus", mock_focus)

        from openbrowser.browser.events import TabClosedEvent

        event = TabClosedEvent(target_id="target-1")
        await session.on_TabClosedEvent(event)
        mock_bus.dispatch.assert_called()

    @pytest.mark.asyncio
    async def test_other_tab_closed_no_switch(self):
        """Test closing a different tab doesn't trigger switch."""
        session = BrowserSession()
        mock_bus = MagicMock()
        mock_bus.dispatch = AsyncMock()
        object.__setattr__(session, "event_bus", mock_bus)
        # Use object.__setattr__ to bypass Pydantic validation for agent_focus
        mock_focus = MagicMock()
        mock_focus.target_id = "target-1"
        object.__setattr__(session, "agent_focus", mock_focus)

        from openbrowser.browser.events import TabClosedEvent

        event = TabClosedEvent(target_id="target-2")
        await session.on_TabClosedEvent(event)
        mock_bus.dispatch.assert_not_called()
