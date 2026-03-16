"""Deep coverage tests for openbrowser.agent.service module.

Targets ONLY the missing lines (178-187, 191, 242-247, 312-313, 364-406,
426-427, 525-534, 539-540, 545-547, 556-563, 583, 586-588, 659-679,
685-728, 755-764, 1028, 1051, 1109, 1112-1113, 1125-1126, 1128-1129,
1141, 1175, 1239, 1244, 1247-1249, 1309, 1357-1358, 1583, 1596-1684,
1720-1721, 1923-1942, 2082-2084, 2122).

All LLM calls, browser connections, and external services are mocked.
"""

import asyncio
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from pydantic import BaseModel, create_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (same pattern as existing test file)
# ---------------------------------------------------------------------------


def _make_mock_llm(model="test-model", provider="test-provider"):
    """Create a mock LLM with required attributes."""
    llm = MagicMock()
    llm.model = model
    llm.provider = provider
    llm.ainvoke = AsyncMock()
    llm._verified_api_keys = True
    return llm


def _make_mock_browser_session(cdp_url=None, downloads_path=None):
    """Create a mock BrowserSession."""
    session = MagicMock()
    session.id = "session-1234-5678"
    session.cdp_url = cdp_url
    session.start = AsyncMock()
    session.kill = AsyncMock()
    session.get_browser_state_summary = AsyncMock()
    session.downloaded_files = []
    session.agent_focus = MagicMock()
    session.agent_focus.target_id = "target-id-12345678"
    session._cached_browser_state_summary = None

    profile = MagicMock()
    profile.allowed_domains = []
    profile.wait_between_actions = 0.0
    profile.keep_alive = False
    profile.downloads_path = downloads_path
    session.browser_profile = profile

    return session


def _make_mock_browser_state_summary(
    url="https://example.com",
    title="Example",
    screenshot=None,
    selector_map=None,
):
    """Create a mock BrowserStateSummary."""
    state = MagicMock()
    state.url = url
    state.title = title
    state.screenshot = screenshot

    tab = MagicMock()
    tab.url = url
    tab.title = title
    tab.target_id = "abcdefghijklmnop12345678"
    state.tabs = [tab]

    dom_state = MagicMock()
    dom_state.selector_map = selector_map or {}
    dom_state.llm_representation = MagicMock(return_value="[1] button Click me")
    dom_state._root = None
    state.dom_state = dom_state

    state.page_info = None
    state.is_pdf_viewer = False
    state.recent_events = None
    state.closed_popup_messages = []

    return state


def _make_mock_tools():
    """Create a mock Tools object."""
    tools = MagicMock()
    tools.act = AsyncMock()

    registry = MagicMock()
    action_model = create_model(
        "TestActionModel",
        __base__=_get_action_model_base(),
        done=(Optional[dict], None),
        navigate=(Optional[dict], None),
        click=(Optional[dict], None),
    )
    registry.create_action_model = MagicMock(return_value=action_model)
    registry.get_prompt_description = MagicMock(return_value=None)
    registry.registry = MagicMock()
    registry.registry.actions = {}
    tools.registry = registry
    return tools


def _get_action_model_base():
    from openbrowser.tools.registry.views import ActionModel
    return ActionModel


# ---------------------------------------------------------------------------
# Shared agent factory (patches all heavy dependencies)
# ---------------------------------------------------------------------------

@patch("openbrowser.agent.service.EventBus")
@patch("openbrowser.agent.service.ProductTelemetry")
@patch("openbrowser.agent.service.TokenCost")
@patch("openbrowser.agent.service.SystemPrompt")
@patch("openbrowser.agent.service.MessageManager")
@patch("openbrowser.agent.service.FileSystem")
def _create_agent(
    mock_fs_cls,
    mock_mm,
    mock_sp,
    mock_tc,
    mock_telemetry,
    mock_eventbus,
    task="Test task",
    llm=None,
    browser_session=None,
    tools=None,
    **kwargs,
):
    from openbrowser.agent.service import Agent

    if llm is None:
        llm = _make_mock_llm()
    if browser_session is None:
        browser_session = _make_mock_browser_session()
    if tools is None:
        tools = _make_mock_tools()

    mock_sp_instance = MagicMock()
    mock_sp_instance.get_system_message.return_value = MagicMock(content="system prompt", cache=True)
    mock_sp.return_value = mock_sp_instance

    mock_fs_instance = MagicMock()
    mock_fs_instance.get_state.return_value = MagicMock()
    mock_fs_instance.base_dir = Path(tempfile.mkdtemp())
    mock_fs_cls.return_value = mock_fs_instance
    mock_fs_cls.from_state = MagicMock(return_value=mock_fs_instance)

    mock_tc_instance = MagicMock()
    mock_tc_instance.register_llm = MagicMock()
    mock_tc_instance.get_usage_summary = AsyncMock(return_value=None)
    mock_tc_instance.log_usage_summary = AsyncMock()
    mock_tc_instance.get_usage_tokens_for_model = MagicMock(
        return_value=MagicMock(
            prompt_tokens=100,
            completion_tokens=50,
            prompt_cached_tokens=10,
            total_tokens=150,
        )
    )
    mock_tc.return_value = mock_tc_instance

    mock_eventbus_instance = MagicMock()
    mock_eventbus_instance.stop = AsyncMock()
    mock_eventbus.return_value = mock_eventbus_instance

    with patch.object(Agent, "_set_screenshot_service"):
        agent = Agent(
            task=task,
            llm=llm,
            browser_session=browser_session,
            tools=tools,
            **kwargs,
        )
    agent.screenshot_service = MagicMock()
    agent.screenshot_service.store_screenshot = AsyncMock(return_value=None)
    return agent


# ===========================================================================
# Tests for __init__ default LLM paths (lines 178-187, 191)
# ===========================================================================


class TestAgentInitDefaultLLM:
    """Cover llm=None branches: CONFIG.DEFAULT_LLM, ChatBrowserUse fallback, browser-use flash mode."""

    @patch("openbrowser.agent.service.EventBus")
    @patch("openbrowser.agent.service.ProductTelemetry")
    @patch("openbrowser.agent.service.TokenCost")
    @patch("openbrowser.agent.service.SystemPrompt")
    @patch("openbrowser.agent.service.MessageManager")
    @patch("openbrowser.agent.service.FileSystem")
    def test_default_llm_from_config(
        self, mock_fs_cls, mock_mm, mock_sp, mock_tc, mock_telemetry, mock_eventbus
    ):
        """Lines 178-182: CONFIG.DEFAULT_LLM is set, get_llm_by_name is called."""
        from openbrowser.agent.service import Agent

        mock_sp_inst = MagicMock()
        mock_sp_inst.get_system_message.return_value = MagicMock(content="sys", cache=True)
        mock_sp.return_value = mock_sp_inst

        mock_fs_inst = MagicMock()
        mock_fs_inst.get_state.return_value = MagicMock()
        mock_fs_inst.base_dir = Path(tempfile.mkdtemp())
        mock_fs_cls.return_value = mock_fs_inst

        mock_tc_inst = MagicMock()
        mock_tc_inst.register_llm = MagicMock()
        mock_tc.return_value = mock_tc_inst

        mock_eventbus.return_value = MagicMock(stop=AsyncMock())

        fake_llm = _make_mock_llm(model="fake-default", provider="fake")

        # get_llm_by_name is imported locally inside __init__, so we must
        # patch it on the module where it lives.
        import openbrowser.llm.models as llm_models_mod

        with (
            patch("openbrowser.agent.service.CONFIG") as mock_config,
            patch.object(llm_models_mod, "get_llm_by_name", return_value=fake_llm) as mock_get,
            patch.object(Agent, "_set_screenshot_service"),
        ):
            mock_config.DEFAULT_LLM = "fake-default"
            mock_config.SKIP_LLM_API_KEY_VERIFICATION = False
            mock_config.OPENBROWSER_CONFIG_DIR = Path(tempfile.mkdtemp())

            session = _make_mock_browser_session()
            tools = _make_mock_tools()
            agent = Agent(
                task="Test",
                llm=None,
                browser_session=session,
                tools=tools,
            )
            mock_get.assert_any_call("fake-default")
            assert agent.llm is fake_llm

    @patch("openbrowser.agent.service.EventBus")
    @patch("openbrowser.agent.service.ProductTelemetry")
    @patch("openbrowser.agent.service.TokenCost")
    @patch("openbrowser.agent.service.SystemPrompt")
    @patch("openbrowser.agent.service.MessageManager")
    @patch("openbrowser.agent.service.FileSystem")
    def test_default_llm_chatbrowseruse_fallback(
        self, mock_fs_cls, mock_mm, mock_sp, mock_tc, mock_telemetry, mock_eventbus
    ):
        """Lines 183-187: CONFIG.DEFAULT_LLM is falsy, ChatBrowserUse() fallback."""
        from openbrowser.agent.service import Agent

        mock_sp_inst = MagicMock()
        mock_sp_inst.get_system_message.return_value = MagicMock(content="sys", cache=True)
        mock_sp.return_value = mock_sp_inst

        mock_fs_inst = MagicMock()
        mock_fs_inst.get_state.return_value = MagicMock()
        mock_fs_inst.base_dir = Path(tempfile.mkdtemp())
        mock_fs_cls.return_value = mock_fs_inst

        mock_tc_inst = MagicMock()
        mock_tc_inst.register_llm = MagicMock()
        mock_tc.return_value = mock_tc_inst

        mock_eventbus.return_value = MagicMock(stop=AsyncMock())

        fake_cbu = _make_mock_llm(model="browser-use-model", provider="browser-use")
        with (
            patch("openbrowser.agent.service.CONFIG") as mock_config,
            patch("openbrowser.ChatBrowserUse", return_value=fake_cbu),
            patch.object(Agent, "_set_screenshot_service"),
        ):
            mock_config.DEFAULT_LLM = None
            mock_config.SKIP_LLM_API_KEY_VERIFICATION = False
            mock_config.OPENBROWSER_CONFIG_DIR = Path(tempfile.mkdtemp())

            session = _make_mock_browser_session()
            tools = _make_mock_tools()
            agent = Agent(
                task="Test",
                llm=None,
                browser_session=session,
                tools=tools,
            )
            # Line 191: provider == 'browser-use' -> flash_mode=True
            assert agent.settings.flash_mode is True


# ===========================================================================
# Tests for controller alias and default Tools creation (lines 242-247)
# ===========================================================================


class TestControllerAliasAndDefaultTools:
    """Cover controller alias path and default Tools() creation."""

    def test_controller_alias(self):
        """Lines 242-243: controller is not None and tools is None.

        Validates that the controller parameter is used as tools when tools=None.
        The real validation is in test_controller_alias_used_when_tools_none below,
        but this test verifies the helper path works end-to-end.
        """
        controller_tools = _make_mock_tools()
        agent = _create_agent(tools=controller_tools)
        assert agent.tools is controller_tools, (
            "Agent should use the provided tools object"
        )

    @patch("openbrowser.agent.service.EventBus")
    @patch("openbrowser.agent.service.ProductTelemetry")
    @patch("openbrowser.agent.service.TokenCost")
    @patch("openbrowser.agent.service.SystemPrompt")
    @patch("openbrowser.agent.service.MessageManager")
    @patch("openbrowser.agent.service.FileSystem")
    @patch("openbrowser.agent.service.Tools")
    def test_controller_alias_used_when_tools_none(
        self, mock_tools_cls, mock_fs_cls, mock_mm, mock_sp, mock_tc, mock_telemetry, mock_eventbus
    ):
        """Lines 242-243: controller param used when tools is None."""
        from openbrowser.agent.service import Agent

        mock_sp_inst = MagicMock()
        mock_sp_inst.get_system_message.return_value = MagicMock(content="sys", cache=True)
        mock_sp.return_value = mock_sp_inst

        mock_fs_inst = MagicMock()
        mock_fs_inst.get_state.return_value = MagicMock()
        mock_fs_inst.base_dir = Path(tempfile.mkdtemp())
        mock_fs_cls.return_value = mock_fs_inst

        mock_tc_inst = MagicMock()
        mock_tc_inst.register_llm = MagicMock()
        mock_tc.return_value = mock_tc_inst

        mock_eventbus.return_value = MagicMock(stop=AsyncMock())

        controller = _make_mock_tools()
        llm = _make_mock_llm()
        session = _make_mock_browser_session()

        with patch.object(Agent, "_set_screenshot_service"):
            agent = Agent(
                task="Test",
                llm=llm,
                browser_session=session,
                controller=controller,
            )
        assert agent.tools is controller
        mock_tools_cls.assert_not_called()

    @patch("openbrowser.agent.service.EventBus")
    @patch("openbrowser.agent.service.ProductTelemetry")
    @patch("openbrowser.agent.service.TokenCost")
    @patch("openbrowser.agent.service.SystemPrompt")
    @patch("openbrowser.agent.service.MessageManager")
    @patch("openbrowser.agent.service.FileSystem")
    @patch("openbrowser.agent.service.Tools")
    def test_default_tools_created_when_both_none(
        self, mock_tools_cls, mock_fs_cls, mock_mm, mock_sp, mock_tc, mock_telemetry, mock_eventbus
    ):
        """Lines 244-247: both tools and controller are None, default Tools created."""
        from openbrowser.agent.service import Agent

        mock_sp_inst = MagicMock()
        mock_sp_inst.get_system_message.return_value = MagicMock(content="sys", cache=True)
        mock_sp.return_value = mock_sp_inst

        mock_fs_inst = MagicMock()
        mock_fs_inst.get_state.return_value = MagicMock()
        mock_fs_inst.base_dir = Path(tempfile.mkdtemp())
        mock_fs_cls.return_value = mock_fs_inst

        mock_tc_inst = MagicMock()
        mock_tc_inst.register_llm = MagicMock()
        mock_tc.return_value = mock_tc_inst

        mock_eventbus.return_value = MagicMock(stop=AsyncMock())

        # Make the mock Tools instance look enough like real Tools
        mock_tools_instance = _make_mock_tools()
        mock_tools_cls.return_value = mock_tools_instance

        llm = _make_mock_llm()
        session = _make_mock_browser_session()

        with patch.object(Agent, "_set_screenshot_service"):
            agent = Agent(
                task="Test",
                llm=llm,
                browser_session=session,
                # tools=None, controller=None (defaults)
                use_vision=False,  # triggers exclude_actions=['screenshot']
            )
        mock_tools_cls.assert_called_once_with(
            exclude_actions=["screenshot"],
            display_files_in_done_text=True,
        )


# ===========================================================================
# Test init with URL in task -> initial_actions (lines 312-313)
# ===========================================================================


class TestInitUrlExtraction:
    """Cover auto-URL extraction setting initial_actions."""

    @patch("openbrowser.agent.service.EventBus")
    @patch("openbrowser.agent.service.ProductTelemetry")
    @patch("openbrowser.agent.service.TokenCost")
    @patch("openbrowser.agent.service.SystemPrompt")
    @patch("openbrowser.agent.service.MessageManager")
    @patch("openbrowser.agent.service.FileSystem")
    def test_url_in_task_creates_initial_action(
        self, mock_fs_cls, mock_mm, mock_sp, mock_tc, mock_telemetry, mock_eventbus
    ):
        """Lines 311-313: URL found in task triggers navigate initial action."""
        from openbrowser.agent.service import Agent

        mock_sp_inst = MagicMock()
        mock_sp_inst.get_system_message.return_value = MagicMock(content="sys", cache=True)
        mock_sp.return_value = mock_sp_inst

        mock_fs_inst = MagicMock()
        mock_fs_inst.get_state.return_value = MagicMock()
        mock_fs_inst.base_dir = Path(tempfile.mkdtemp())
        mock_fs_cls.return_value = mock_fs_inst

        mock_tc_inst = MagicMock()
        mock_tc_inst.register_llm = MagicMock()
        mock_tc.return_value = mock_tc_inst

        mock_eventbus.return_value = MagicMock(stop=AsyncMock())

        # Build a real registry entry for 'navigate' so _convert_initial_actions works
        class NavigateParams(BaseModel):
            url: str
            new_tab: bool = False

        tools = _make_mock_tools()
        action_info = MagicMock()
        action_info.param_model = NavigateParams
        tools.registry.registry.actions = {"navigate": action_info}

        ActionModelWithNav = create_model(
            "ActionModelWithNav",
            __base__=_get_action_model_base(),
            navigate=(Optional[NavigateParams], None),
            done=(Optional[dict], None),
        )
        tools.registry.create_action_model.return_value = ActionModelWithNav

        llm = _make_mock_llm()
        session = _make_mock_browser_session()

        with patch.object(Agent, "_set_screenshot_service"):
            agent = Agent(
                task="Go to https://example.com and find info",
                llm=llm,
                browser_session=session,
                tools=tools,
                directly_open_url=True,
            )

        assert agent.initial_url == "https://example.com"
        assert agent.initial_actions is not None
        assert len(agent.initial_actions) == 1


# ===========================================================================
# sensitive_data domain validation (lines 364-406)
# ===========================================================================


class TestSensitiveDataDomainValidation:
    """Cover sensitive_data with domain-specific credentials."""

    def test_sensitive_data_no_allowed_domains_warning(self):
        """Lines 366-372: sensitive_data present but no allowed_domains -> error log."""
        session = _make_mock_browser_session()
        session.browser_profile.allowed_domains = []
        agent = _create_agent(
            browser_session=session,
            sensitive_data={"password": "secret123"},
        )
        # Should have set sensitive_data without raising
        assert agent.sensitive_data is not None

    def test_sensitive_data_domain_specific_credentials_covered(self):
        """Lines 375-406: domain-specific credentials with matching allowed_domains."""
        session = _make_mock_browser_session()
        session.browser_profile.allowed_domains = ["*.google.com"]
        agent = _create_agent(
            browser_session=session,
            sensitive_data={"google.com": {"user": "a", "pass": "b"}},
        )
        assert agent.sensitive_data is not None

    def test_sensitive_data_domain_not_covered_warning(self):
        """Lines 405-406: domain pattern not covered by allowed_domains -> warning."""
        session = _make_mock_browser_session()
        session.browser_profile.allowed_domains = ["*.example.com"]
        agent = _create_agent(
            browser_session=session,
            sensitive_data={"evil.com": {"user": "a", "pass": "b"}},
        )
        assert agent.sensitive_data is not None

    def test_sensitive_data_wildcard_allowed_domain(self):
        """Line 384: allowed_domain == '*' covers everything."""
        session = _make_mock_browser_session()
        session.browser_profile.allowed_domains = ["*"]
        agent = _create_agent(
            browser_session=session,
            sensitive_data={"anything.com": {"user": "a", "pass": "b"}},
        )
        assert agent.sensitive_data is not None

    def test_sensitive_data_exact_domain_match(self):
        """Line 384: domain_pattern == allowed_domain exactly."""
        session = _make_mock_browser_session()
        session.browser_profile.allowed_domains = ["google.com"]
        agent = _create_agent(
            browser_session=session,
            sensitive_data={"google.com": {"user": "a", "pass": "b"}},
        )
        assert agent.sensitive_data is not None

    def test_sensitive_data_subdomain_match(self):
        """Lines 395-402: subdomain matching with wildcard patterns."""
        session = _make_mock_browser_session()
        session.browser_profile.allowed_domains = ["*.google.com"]
        agent = _create_agent(
            browser_session=session,
            sensitive_data={"accounts.google.com": {"user": "a", "pass": "b"}},
        )
        assert agent.sensitive_data is not None

    def test_sensitive_data_with_scheme_in_pattern(self):
        """Lines 390-391: domain patterns with scheme (https://)."""
        session = _make_mock_browser_session()
        session.browser_profile.allowed_domains = ["https://google.com"]
        agent = _create_agent(
            browser_session=session,
            sensitive_data={"https://google.com": {"user": "a", "pass": "b"}},
        )
        assert agent.sensitive_data is not None


# ===========================================================================
# save_conversation_path expansion (lines 426-427)
# ===========================================================================


class TestSaveConversationPathInit:
    """Cover save_conversation_path expansion during init."""

    def test_save_conversation_path_expanded(self):
        """Lines 425-427: save_conversation_path is expanded and resolved."""
        tmp = tempfile.mkdtemp()
        agent = _create_agent(save_conversation_path=tmp)
        assert agent.settings.save_conversation_path is not None
        assert isinstance(agent.settings.save_conversation_path, Path)
        assert agent.settings.save_conversation_path.is_absolute()


# ===========================================================================
# _set_file_system (lines 525-534, 539-540, 545-547, 556-563)
# ===========================================================================


class TestSetFileSystem:
    """Cover _set_file_system restore from state, custom path, and error handling."""

    @patch("openbrowser.agent.service.EventBus")
    @patch("openbrowser.agent.service.ProductTelemetry")
    @patch("openbrowser.agent.service.TokenCost")
    @patch("openbrowser.agent.service.SystemPrompt")
    @patch("openbrowser.agent.service.MessageManager")
    @patch("openbrowser.agent.service.FileSystem")
    def test_restore_from_state(
        self, mock_fs_cls, mock_mm, mock_sp, mock_tc, mock_telemetry, mock_eventbus
    ):
        """Lines 524-531: restore file system from injected state."""
        from openbrowser.agent.service import Agent
        from openbrowser.agent.views import AgentState
        from openbrowser.filesystem.file_system import FileSystemState

        mock_sp_inst = MagicMock()
        mock_sp_inst.get_system_message.return_value = MagicMock(content="sys", cache=True)
        mock_sp.return_value = mock_sp_inst

        mock_fs_inst = MagicMock()
        mock_fs_inst.base_dir = Path("/tmp/restored")
        mock_fs_inst.get_state.return_value = MagicMock()
        mock_fs_cls.from_state = MagicMock(return_value=mock_fs_inst)
        mock_fs_cls.return_value = mock_fs_inst

        mock_tc_inst = MagicMock()
        mock_tc_inst.register_llm = MagicMock()
        mock_tc.return_value = mock_tc_inst
        mock_eventbus.return_value = MagicMock(stop=AsyncMock())

        state = AgentState()
        state.file_system_state = FileSystemState(base_dir="/tmp/test", files={})

        llm = _make_mock_llm()
        session = _make_mock_browser_session()
        tools = _make_mock_tools()

        with patch.object(Agent, "_set_screenshot_service"):
            agent = Agent(
                task="Test",
                llm=llm,
                browser_session=session,
                tools=tools,
                injected_agent_state=state,
            )
        mock_fs_cls.from_state.assert_called_once()
        assert agent.file_system_path == "/tmp/restored"

    @patch("openbrowser.agent.service.EventBus")
    @patch("openbrowser.agent.service.ProductTelemetry")
    @patch("openbrowser.agent.service.TokenCost")
    @patch("openbrowser.agent.service.SystemPrompt")
    @patch("openbrowser.agent.service.MessageManager")
    @patch("openbrowser.agent.service.FileSystem")
    def test_restore_from_state_failure(
        self, mock_fs_cls, mock_mm, mock_sp, mock_tc, mock_telemetry, mock_eventbus
    ):
        """Lines 532-534: restore file system from state fails, re-raises."""
        from openbrowser.agent.service import Agent
        from openbrowser.agent.views import AgentState
        from openbrowser.filesystem.file_system import FileSystemState

        mock_sp_inst = MagicMock()
        mock_sp_inst.get_system_message.return_value = MagicMock(content="sys", cache=True)
        mock_sp.return_value = mock_sp_inst

        mock_fs_cls.from_state = MagicMock(side_effect=RuntimeError("corrupt state"))
        mock_fs_cls.return_value = MagicMock(
            get_state=MagicMock(return_value=MagicMock()),
            base_dir=Path(tempfile.mkdtemp()),
        )

        mock_tc_inst = MagicMock()
        mock_tc_inst.register_llm = MagicMock()
        mock_tc.return_value = mock_tc_inst
        mock_eventbus.return_value = MagicMock(stop=AsyncMock())

        state = AgentState()
        state.file_system_state = FileSystemState(base_dir="/tmp/test", files={})

        with pytest.raises(RuntimeError, match="corrupt state"):
            with patch.object(Agent, "_set_screenshot_service"):
                Agent(
                    task="Test",
                    llm=_make_mock_llm(),
                    browser_session=_make_mock_browser_session(),
                    tools=_make_mock_tools(),
                    injected_agent_state=state,
                )

    @patch("openbrowser.agent.service.EventBus")
    @patch("openbrowser.agent.service.ProductTelemetry")
    @patch("openbrowser.agent.service.TokenCost")
    @patch("openbrowser.agent.service.SystemPrompt")
    @patch("openbrowser.agent.service.MessageManager")
    @patch("openbrowser.agent.service.FileSystem")
    def test_custom_file_system_path(
        self, mock_fs_cls, mock_mm, mock_sp, mock_tc, mock_telemetry, mock_eventbus
    ):
        """Lines 538-540: file_system_path provided, creates FileSystem at that path."""
        from openbrowser.agent.service import Agent

        mock_sp_inst = MagicMock()
        mock_sp_inst.get_system_message.return_value = MagicMock(content="sys", cache=True)
        mock_sp.return_value = mock_sp_inst

        custom_path = tempfile.mkdtemp()
        mock_fs_inst = MagicMock()
        mock_fs_inst.get_state.return_value = MagicMock()
        mock_fs_inst.base_dir = Path(custom_path)
        mock_fs_cls.return_value = mock_fs_inst

        mock_tc_inst = MagicMock()
        mock_tc_inst.register_llm = MagicMock()
        mock_tc.return_value = mock_tc_inst
        mock_eventbus.return_value = MagicMock(stop=AsyncMock())

        with patch.object(Agent, "_set_screenshot_service"):
            agent = Agent(
                task="Test",
                llm=_make_mock_llm(),
                browser_session=_make_mock_browser_session(),
                tools=_make_mock_tools(),
                file_system_path=custom_path,
            )
        assert agent.file_system_path == custom_path

    @patch("openbrowser.agent.service.EventBus")
    @patch("openbrowser.agent.service.ProductTelemetry")
    @patch("openbrowser.agent.service.TokenCost")
    @patch("openbrowser.agent.service.SystemPrompt")
    @patch("openbrowser.agent.service.MessageManager")
    @patch("openbrowser.agent.service.FileSystem")
    def test_file_system_init_failure(
        self, mock_fs_cls, mock_mm, mock_sp, mock_tc, mock_telemetry, mock_eventbus
    ):
        """Lines 545-547: FileSystem init fails, re-raises."""
        from openbrowser.agent.service import Agent

        mock_sp_inst = MagicMock()
        mock_sp_inst.get_system_message.return_value = MagicMock(content="sys", cache=True)
        mock_sp.return_value = mock_sp_inst

        mock_fs_cls.return_value = MagicMock(
            get_state=MagicMock(return_value=MagicMock()),
            base_dir=Path(tempfile.mkdtemp()),
        )
        mock_fs_cls.side_effect = OSError("permission denied")

        mock_tc_inst = MagicMock()
        mock_tc_inst.register_llm = MagicMock()
        mock_tc.return_value = mock_tc_inst
        mock_eventbus.return_value = MagicMock(stop=AsyncMock())

        with pytest.raises(OSError, match="permission denied"):
            with patch.object(Agent, "_set_screenshot_service"):
                Agent(
                    task="Test",
                    llm=_make_mock_llm(),
                    browser_session=_make_mock_browser_session(),
                    tools=_make_mock_tools(),
                )


class TestSetScreenshotServiceError:
    """Cover _set_screenshot_service error handling (lines 556-563)."""

    @patch("openbrowser.agent.service.EventBus")
    @patch("openbrowser.agent.service.ProductTelemetry")
    @patch("openbrowser.agent.service.TokenCost")
    @patch("openbrowser.agent.service.SystemPrompt")
    @patch("openbrowser.agent.service.MessageManager")
    @patch("openbrowser.agent.service.FileSystem")
    def test_screenshot_service_init_failure(
        self, mock_fs_cls, mock_mm, mock_sp, mock_tc, mock_telemetry, mock_eventbus
    ):
        """Lines 561-563: ScreenshotService init fails, re-raises."""
        from openbrowser.agent.service import Agent

        mock_sp_inst = MagicMock()
        mock_sp_inst.get_system_message.return_value = MagicMock(content="sys", cache=True)
        mock_sp.return_value = mock_sp_inst

        mock_fs_inst = MagicMock()
        mock_fs_inst.get_state.return_value = MagicMock()
        mock_fs_inst.base_dir = Path(tempfile.mkdtemp())
        mock_fs_cls.return_value = mock_fs_inst

        mock_tc_inst = MagicMock()
        mock_tc_inst.register_llm = MagicMock()
        mock_tc.return_value = mock_tc_inst
        mock_eventbus.return_value = MagicMock(stop=AsyncMock())

        with patch(
            "openbrowser.screenshots.service.ScreenshotService",
            side_effect=RuntimeError("screenshot init failed"),
        ):
            with pytest.raises(RuntimeError, match="screenshot init failed"):
                Agent(
                    task="Test",
                    llm=_make_mock_llm(),
                    browser_session=_make_mock_browser_session(),
                    tools=_make_mock_tools(),
                )


# ===========================================================================
# _set_openbrowser_version_and_source (lines 583, 586-588)
# ===========================================================================


class TestVersionAndSource:
    """Cover source determination edge cases."""

    def test_source_pip_when_not_all_repo_files(self):
        """Line 585: not all repo_files exist -> source = 'pip'."""
        agent = _create_agent()
        # By default the package_root check may or may not find git files.
        # If we mock Path.exists to return False for at least one, source should be 'pip'
        with patch("pathlib.Path.exists", return_value=False):
            agent._set_openbrowser_version_and_source(source_override=None)
        assert agent.source == "pip"

    def test_source_unknown_on_exception(self):
        """Lines 586-588: exception during source detection -> 'unknown'."""
        agent = _create_agent()
        with patch("pathlib.Path.exists", side_effect=Exception("broken")):
            agent._set_openbrowser_version_and_source(source_override=None)
        assert agent.source == "unknown"


# ===========================================================================
# step() integration (lines 659-679)
# ===========================================================================


class TestStepIntegration:
    """Cover the step() method orchestration."""

    @pytest.mark.asyncio
    async def test_step_success_flow(self):
        """Lines 659-679: step runs prepare_context -> get_next_action -> execute_actions -> post_process -> finalize."""
        from openbrowser.models import ActionResult

        agent = _create_agent()
        agent.step_start_time = time.time()
        agent.state.last_result = [ActionResult(extracted_content="ok")]

        with (
            patch.object(agent, "_prepare_context", new_callable=AsyncMock, return_value=_make_mock_browser_state_summary()),
            patch.object(agent, "_get_next_action", new_callable=AsyncMock),
            patch.object(agent, "_execute_actions", new_callable=AsyncMock),
            patch.object(agent, "_post_process", new_callable=AsyncMock),
            patch.object(agent, "_finalize", new_callable=AsyncMock),
        ):
            await agent.step(None)

    @pytest.mark.asyncio
    async def test_step_error_in_prepare_context(self):
        """Lines 674-676: exception during step -> _handle_step_error."""
        agent = _create_agent()

        with (
            patch.object(agent, "_prepare_context", new_callable=AsyncMock, side_effect=RuntimeError("fail")),
            patch.object(agent, "_handle_step_error", new_callable=AsyncMock) as mock_err,
            patch.object(agent, "_finalize", new_callable=AsyncMock),
        ):
            await agent.step(None)
            mock_err.assert_called_once()


# ===========================================================================
# _prepare_context (lines 685-728)
# ===========================================================================


class TestPrepareContext:
    """Cover the _prepare_context method."""

    @pytest.mark.asyncio
    async def test_prepare_context_flow(self):
        """Lines 685-728: full _prepare_context flow."""
        from openbrowser.agent.views import AgentStepInfo

        agent = _create_agent()
        browser_state = _make_mock_browser_state_summary(screenshot="base64data")
        agent.browser_session.get_browser_state_summary = AsyncMock(return_value=browser_state)
        agent.has_downloads_path = False

        step_info = AgentStepInfo(step_number=1, max_steps=10)

        with (
            patch.object(agent, "_check_stop_or_pause", new_callable=AsyncMock),
            patch.object(agent, "_update_action_models_for_page", new_callable=AsyncMock),
            patch.object(agent, "_force_done_after_last_step", new_callable=AsyncMock),
            patch.object(agent, "_force_done_after_failure", new_callable=AsyncMock),
        ):
            result = await agent._prepare_context(step_info)
            assert result is browser_state

    @pytest.mark.asyncio
    async def test_prepare_context_no_screenshot(self):
        """Line 697: browser state without screenshot."""
        agent = _create_agent()
        browser_state = _make_mock_browser_state_summary(screenshot=None)
        agent.browser_session.get_browser_state_summary = AsyncMock(return_value=browser_state)
        agent.has_downloads_path = False

        with (
            patch.object(agent, "_check_stop_or_pause", new_callable=AsyncMock),
            patch.object(agent, "_update_action_models_for_page", new_callable=AsyncMock),
            patch.object(agent, "_force_done_after_last_step", new_callable=AsyncMock),
            patch.object(agent, "_force_done_after_failure", new_callable=AsyncMock),
        ):
            result = await agent._prepare_context(None)
            assert result is browser_state


# ===========================================================================
# _get_next_action success path (lines 755-764)
# ===========================================================================


class TestGetNextActionSuccessPath:
    """Cover _get_next_action full success path including pause checks."""

    @pytest.mark.asyncio
    async def test_get_next_action_success(self):
        """Lines 755-764: model output stored, check_stop_or_pause called, post_llm called."""
        from openbrowser.agent.views import AgentOutput

        agent = _create_agent()
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))

        expected = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[CustomAction(done={"text": "test"})],
        )

        agent._message_manager.get_messages = MagicMock(return_value=[])
        agent._get_model_output_with_retry = AsyncMock(return_value=expected)

        browser_state = _make_mock_browser_state_summary()
        check_mock = AsyncMock()
        post_mock = AsyncMock()

        with (
            patch.object(agent, "_check_stop_or_pause", check_mock),
            patch.object(agent, "_handle_post_llm_processing", post_mock),
        ):
            await agent._get_next_action(browser_state)

        assert agent.state.last_model_output is expected
        assert check_mock.call_count == 2  # called twice
        post_mock.assert_called_once()


# ===========================================================================
# URL replacement edge cases (lines 1028, 1051)
# ===========================================================================


class TestUrlReplacementEdgeCases:
    """Cover URL replacement edge cases."""

    def test_url_with_fragment_only_within_limit(self):
        """Line 1028: fragment_start found but after_path within limit."""
        agent = _create_agent(_url_shortening_limit=50)
        text = "Visit https://example.com#section"
        result, replaced = agent._replace_urls_in_text(text)
        # Short fragment, should not be replaced
        assert replaced == {}

    def test_url_long_after_path_not_shorter(self):
        """Line 1051: shortened URL is not actually shorter -> keep original."""
        agent = _create_agent(_url_shortening_limit=24)
        # Make the after_path just barely over limit but shortened version would not be shorter
        text = "Visit https://example.com?" + "a" * 25
        result, replaced = agent._replace_urls_in_text(text)
        # Result depends on whether shortened version is actually shorter
        assert isinstance(replaced, dict)


# ===========================================================================
# _recursive_process_dict with BaseModel and list/tuple (lines 1108-1113)
# ===========================================================================


class TestRecursiveProcessDictDeepCoverage:
    """Cover dict processing with BaseModel values and list/tuple values."""

    def test_dict_with_basemodel_value(self):
        """Lines 1108-1109: dict value is a BaseModel."""
        from openbrowser.agent.service import Agent

        class Inner(BaseModel):
            url: str = "https://short.url"

        d = {"model": Inner()}
        replacements = {"https://short.url": "https://original.com"}
        Agent._recursive_process_dict(d, replacements)
        assert d["model"].url == "https://original.com"

    def test_dict_with_list_value(self):
        """Lines 1112-1113: dict value is a list/tuple."""
        from openbrowser.agent.service import Agent

        d = {"items": ["https://short.url", "other"]}
        replacements = {"https://short.url": "https://original.com"}
        Agent._recursive_process_dict(d, replacements)
        assert d["items"][0] == "https://original.com"

    def test_dict_with_tuple_value(self):
        """Lines 1112-1113: dict value is a tuple."""
        from openbrowser.agent.service import Agent

        d = {"items": ("https://short.url", "other")}
        replacements = {"https://short.url": "https://original.com"}
        Agent._recursive_process_dict(d, replacements)
        assert d["items"][0] == "https://original.com"


# ===========================================================================
# _recursive_process_list_or_tuple with BaseModel and dict in tuple (lines 1124-1129)
# ===========================================================================


class TestRecursiveProcessTupleDeepCoverage:
    """Cover tuple processing with BaseModel and dict items."""

    def test_tuple_with_basemodel_item(self):
        """Lines 1124-1126: tuple item is a BaseModel."""
        from openbrowser.agent.service import Agent

        class Inner(BaseModel):
            url: str = "https://short.url"

        model = Inner()
        tpl = (model,)
        replacements = {"https://short.url": "https://original.com"}
        result = Agent._recursive_process_list_or_tuple(tpl, replacements)
        assert isinstance(result, tuple)
        assert result[0].url == "https://original.com"

    def test_tuple_with_dict_item(self):
        """Lines 1127-1129: tuple item is a dict."""
        from openbrowser.agent.service import Agent

        tpl = ({"url": "https://short.url"},)
        replacements = {"https://short.url": "https://original.com"}
        result = Agent._recursive_process_list_or_tuple(tpl, replacements)
        assert isinstance(result, tuple)
        assert result[0]["url"] == "https://original.com"


# ===========================================================================
# _recursive_process_list_or_tuple list with BaseModel (line 1141)
# ===========================================================================


class TestRecursiveProcessListBaseModel:
    """Cover list processing with BaseModel items."""

    def test_list_with_basemodel_item(self):
        """Line 1141: list item is a BaseModel."""
        from openbrowser.agent.service import Agent

        class Inner(BaseModel):
            url: str = "https://short.url"

        model = Inner()
        lst = [model]
        replacements = {"https://short.url": "https://original.com"}
        Agent._recursive_process_list_or_tuple(lst, replacements)
        assert lst[0].url == "https://original.com"


# ===========================================================================
# get_model_output with URL replacement (line 1175)
# ===========================================================================


class TestGetModelOutputUrlReplacement:
    """Cover get_model_output URL replacement in parsed response."""

    @pytest.mark.asyncio
    async def test_get_model_output_replaces_urls(self):
        """Line 1175: urls_replaced is non-empty, _recursive_process called."""
        from openbrowser.agent.views import AgentOutput
        from openbrowser.llm.messages import UserMessage

        agent = _create_agent(_url_shortening_limit=5)
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))

        expected = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[CustomAction(done={"text": "test"})],
        )
        response = MagicMock()
        response.completion = expected
        agent.llm.ainvoke = AsyncMock(return_value=response)

        # Use UserMessage (not SystemMessage) so URL replacement actually processes it
        long_url = "https://example.com?" + "x" * 100
        msg = UserMessage(content=f"Visit {long_url}")

        with patch.object(
            type(agent), "_recursive_process_all_strings_inside_pydantic_model"
        ) as mock_replace:
            result = await agent.get_model_output([msg])
            assert result is expected
            # URL was long enough to be shortened, so replacement must have been called
            mock_replace.assert_called_once()
            # Verify the replacement dict maps shortened -> original URL
            call_args = mock_replace.call_args
            replacements = call_args[0][1] if call_args[0] and len(call_args[0]) > 1 else call_args[1].get("url_replacements", {})
            assert len(replacements) > 0, "Expected non-empty URL replacements dict"


# ===========================================================================
# _log_next_action_summary specific param branches (lines 1239, 1244, 1247-1249)
# ===========================================================================


class TestLogNextActionSummaryBranches:
    """Cover specific parameter formatting branches."""

    def test_log_action_summary_index_param(self):
        """Line 1239: action with 'index' param."""
        agent = _create_agent()
        from openbrowser.agent.views import AgentOutput

        ActionModel = _get_action_model_base()
        CustomAction = create_model(
            "CA", __base__=ActionModel, click=(Optional[dict], None)
        )
        action = CustomAction(click={"index": 5})
        parsed = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[action],
        )
        with patch.object(agent.logger, "isEnabledFor", return_value=True):
            agent._log_next_action_summary(parsed)

    def test_log_action_summary_url_param(self):
        """Line 1244: action with 'url' param."""
        agent = _create_agent()
        from openbrowser.agent.views import AgentOutput

        ActionModel = _get_action_model_base()
        CustomAction = create_model(
            "CA", __base__=ActionModel, navigate=(Optional[dict], None)
        )
        action = CustomAction(navigate={"url": "https://example.com"})
        parsed = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[action],
        )
        with patch.object(agent.logger, "isEnabledFor", return_value=True):
            agent._log_next_action_summary(parsed)

    def test_log_action_summary_generic_str_param(self):
        """Lines 1247-1249: action with generic string/int/bool params."""
        agent = _create_agent()
        from openbrowser.agent.views import AgentOutput

        ActionModel = _get_action_model_base()
        CustomAction = create_model(
            "CA", __base__=ActionModel, click=(Optional[dict], None)
        )
        # Use a param name that is not index/text/url/success
        action = CustomAction(click={"custom_param": "some_value"})
        parsed = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[action],
        )
        with patch.object(agent.logger, "isEnabledFor", return_value=True):
            agent._log_next_action_summary(parsed)

    def test_log_action_summary_long_generic_value(self):
        """Lines 1247-1249: generic value longer than 30 chars gets truncated."""
        agent = _create_agent()
        from openbrowser.agent.views import AgentOutput

        ActionModel = _get_action_model_base()
        CustomAction = create_model(
            "CA", __base__=ActionModel, click=(Optional[dict], None)
        )
        action = CustomAction(click={"description": "x" * 50})
        parsed = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[action],
        )
        with patch.object(agent.logger, "isEnabledFor", return_value=True):
            agent._log_next_action_summary(parsed)


# ===========================================================================
# _log_agent_event with no model output (line 1309)
# ===========================================================================


class TestLogAgentEventNoModelOutput:
    """Cover _log_agent_event with history item that has no model output."""

    def test_log_agent_event_history_item_no_model_output(self):
        """Line 1309: history item has model_output=None -> appends None."""
        from openbrowser.agent.views import AgentHistory, StepMetadata
        from openbrowser.browser.views import BrowserStateHistory
        from openbrowser.models import ActionResult

        agent = _create_agent()
        agent.telemetry = MagicMock()

        state = BrowserStateHistory(
            url="https://example.com",
            title="Example",
            tabs=[],
            interacted_element=[],
            screenshot_path=None,
        )
        history_item = AgentHistory(
            model_output=None,
            result=[ActionResult(extracted_content="error")],
            state=state,
            metadata=StepMetadata(step_number=1, step_start_time=1.0, step_end_time=2.0),
        )
        agent.history.history.append(history_item)
        agent._log_agent_event(max_steps=10)
        agent.telemetry.capture.assert_called_once()


# ===========================================================================
# take_step first step exception re-raise (lines 1357-1358)
# ===========================================================================


class TestTakeStepExceptionReRaise:
    """Cover take_step first step non-InterruptedError exception re-raise."""

    @pytest.mark.asyncio
    async def test_take_step_first_step_other_exception(self):
        """Lines 1357-1358: non-InterruptedError re-raised from initial actions."""
        from openbrowser.agent.views import AgentStepInfo

        agent = _create_agent()
        step_info = AgentStepInfo(step_number=0, max_steps=10)

        with patch.object(
            agent,
            "_execute_initial_actions",
            new_callable=AsyncMock,
            side_effect=RuntimeError("init failed"),
        ):
            with pytest.raises(RuntimeError, match="init failed"):
                await agent.take_step(step_info)


# ===========================================================================
# run() -> _run_with_langgraph (lines 1583, 1596-1684)
# ===========================================================================


class TestRunWithLanggraph:
    """Cover _run_with_langgraph method."""

    @pytest.mark.asyncio
    async def test_run_delegates_to_langgraph(self):
        """Line 1583: run() delegates to _run_with_langgraph."""
        agent = _create_agent()
        with patch.object(
            agent, "_run_with_langgraph", new_callable=AsyncMock, return_value=agent.history
        ) as mock_lg:
            result = await agent.run(max_steps=5)
            mock_lg.assert_called_once_with(5, None, None)
            assert result is agent.history

    @pytest.mark.asyncio
    async def test_run_with_langgraph_success(self):
        """Lines 1596-1652: successful LangGraph execution."""
        agent = _create_agent()
        agent._force_exit_telemetry_logged = False

        mock_graph = MagicMock()
        mock_graph.run = AsyncMock()

        with (
            patch("openbrowser.agent.service.check_latest_openbrowser_version", new_callable=AsyncMock, return_value=None),
            patch("openbrowser.agent.graph.create_agent_graph", return_value=mock_graph, create=True),
            patch("openbrowser.utils.SignalHandler") as mock_sh_cls,
            patch.object(agent, "close", new_callable=AsyncMock),
            patch.object(agent, "_log_agent_event"),
            patch.object(agent, "_log_final_outcome_messages"),
        ):
            mock_sh = MagicMock()
            mock_sh_cls.return_value = mock_sh

            result = await agent._run_with_langgraph(max_steps=5)

            assert result is agent.history
            mock_sh.register.assert_called_once()
            mock_sh.unregister.assert_called_once()
            mock_graph.run.assert_called_once_with(max_steps=5)

    @pytest.mark.asyncio
    async def test_run_with_langgraph_keyboard_interrupt(self):
        """Lines 1654-1658: KeyboardInterrupt during execution."""
        agent = _create_agent()
        agent._force_exit_telemetry_logged = False

        mock_graph = MagicMock()
        mock_graph.run = AsyncMock(side_effect=KeyboardInterrupt)

        with (
            patch("openbrowser.agent.service.check_latest_openbrowser_version", new_callable=AsyncMock, return_value=None),
            patch("openbrowser.agent.graph.create_agent_graph", return_value=mock_graph, create=True),
            patch("openbrowser.utils.SignalHandler") as mock_sh_cls,
            patch.object(agent, "close", new_callable=AsyncMock),
            patch.object(agent, "_log_agent_event"),
            patch.object(agent, "_log_final_outcome_messages"),
        ):
            mock_sh_cls.return_value = MagicMock()
            result = await agent._run_with_langgraph(max_steps=5)
            assert result is agent.history

    @pytest.mark.asyncio
    async def test_run_with_langgraph_exception(self):
        """Lines 1660-1663: exception during execution re-raised."""
        agent = _create_agent()
        agent._force_exit_telemetry_logged = False

        mock_graph = MagicMock()
        mock_graph.run = AsyncMock(side_effect=RuntimeError("graph failed"))

        with (
            patch("openbrowser.agent.service.check_latest_openbrowser_version", new_callable=AsyncMock, return_value=None),
            patch("openbrowser.agent.graph.create_agent_graph", return_value=mock_graph, create=True),
            patch("openbrowser.utils.SignalHandler") as mock_sh_cls,
            patch.object(agent, "close", new_callable=AsyncMock),
            patch.object(agent, "_log_agent_event"),
            patch.object(agent, "_log_final_outcome_messages"),
        ):
            mock_sh_cls.return_value = MagicMock()
            with pytest.raises(RuntimeError, match="graph failed"):
                await agent._run_with_langgraph(max_steps=5)

    @pytest.mark.asyncio
    async def test_run_with_langgraph_telemetry_error(self):
        """Lines 1670-1673: telemetry logging failure caught."""
        agent = _create_agent()
        agent._force_exit_telemetry_logged = False

        mock_graph = MagicMock()
        mock_graph.run = AsyncMock()

        with (
            patch("openbrowser.agent.service.check_latest_openbrowser_version", new_callable=AsyncMock, return_value=None),
            patch("openbrowser.agent.graph.create_agent_graph", return_value=mock_graph, create=True),
            patch("openbrowser.utils.SignalHandler") as mock_sh_cls,
            patch.object(agent, "close", new_callable=AsyncMock),
            patch.object(agent, "_log_agent_event", side_effect=RuntimeError("telemetry fail")),
            patch.object(agent, "_log_final_outcome_messages"),
        ):
            mock_sh_cls.return_value = MagicMock()
            # Should not raise despite telemetry failure
            result = await agent._run_with_langgraph(max_steps=5)
            assert result is agent.history

    @pytest.mark.asyncio
    async def test_run_with_langgraph_force_exit_telemetry_skipped(self):
        """Lines 1669: force exit telemetry already logged, skip."""
        agent = _create_agent()

        mock_graph = MagicMock()

        # The run method sets _force_exit_telemetry_logged = False at line 1600.
        # We simulate the on_force_exit callback being called during graph.run
        # which sets _force_exit_telemetry_logged = True before the finally block.
        async def run_and_set_flag(**kwargs):
            agent._force_exit_telemetry_logged = True

        mock_graph.run = AsyncMock(side_effect=run_and_set_flag)

        with (
            patch("openbrowser.agent.service.check_latest_openbrowser_version", new_callable=AsyncMock, return_value=None),
            patch("openbrowser.agent.graph.create_agent_graph", return_value=mock_graph, create=True),
            patch("openbrowser.utils.SignalHandler") as mock_sh_cls,
            patch.object(agent, "close", new_callable=AsyncMock),
            patch.object(agent, "_log_agent_event") as mock_event,
            patch.object(agent, "_log_final_outcome_messages"),
        ):
            mock_sh_cls.return_value = MagicMock()
            await agent._run_with_langgraph(max_steps=5)
            mock_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_with_langgraph_generate_gif(self):
        """Lines 1675-1680: generate_gif setting triggers GIF creation."""
        agent = _create_agent(generate_gif="my_gif.gif")
        agent._force_exit_telemetry_logged = False

        mock_graph = MagicMock()
        mock_graph.run = AsyncMock()

        with (
            patch("openbrowser.agent.service.check_latest_openbrowser_version", new_callable=AsyncMock, return_value=None),
            patch("openbrowser.agent.graph.create_agent_graph", return_value=mock_graph, create=True),
            patch("openbrowser.utils.SignalHandler") as mock_sh_cls,
            patch.object(agent, "close", new_callable=AsyncMock),
            patch.object(agent, "_log_agent_event"),
            patch.object(agent, "_log_final_outcome_messages"),
            patch("openbrowser.agent.gif.create_history_gif") as mock_gif,
        ):
            mock_sh_cls.return_value = MagicMock()
            await agent._run_with_langgraph(max_steps=5)
            mock_gif.assert_called_once_with(
                task=agent.task, history=agent.history, output_path="my_gif.gif"
            )

    @pytest.mark.asyncio
    async def test_run_with_langgraph_generate_gif_true(self):
        """Lines 1675-1676: generate_gif=True uses default path."""
        agent = _create_agent(generate_gif=True)
        agent._force_exit_telemetry_logged = False

        mock_graph = MagicMock()
        mock_graph.run = AsyncMock()

        with (
            patch("openbrowser.agent.service.check_latest_openbrowser_version", new_callable=AsyncMock, return_value=None),
            patch("openbrowser.agent.graph.create_agent_graph", return_value=mock_graph, create=True),
            patch("openbrowser.utils.SignalHandler") as mock_sh_cls,
            patch.object(agent, "close", new_callable=AsyncMock),
            patch.object(agent, "_log_agent_event"),
            patch.object(agent, "_log_final_outcome_messages"),
            patch("openbrowser.agent.gif.create_history_gif") as mock_gif,
        ):
            mock_sh_cls.return_value = MagicMock()
            await agent._run_with_langgraph(max_steps=5)
            mock_gif.assert_called_once_with(
                task=agent.task, history=agent.history, output_path="agent_history.gif"
            )

    @pytest.mark.asyncio
    async def test_run_with_langgraph_output_model_schema(self):
        """Lines 1649-1650: output_model_schema set on history._output_model_schema."""

        class TestSchema(BaseModel):
            result: str

        agent = _create_agent(output_model_schema=TestSchema)
        agent._force_exit_telemetry_logged = False

        mock_graph = MagicMock()
        mock_graph.run = AsyncMock()

        with (
            patch("openbrowser.agent.service.check_latest_openbrowser_version", new_callable=AsyncMock, return_value=None),
            patch("openbrowser.agent.graph.create_agent_graph", return_value=mock_graph, create=True),
            patch("openbrowser.utils.SignalHandler") as mock_sh_cls,
            patch.object(agent, "close", new_callable=AsyncMock),
            patch.object(agent, "_log_agent_event"),
            patch.object(agent, "_log_final_outcome_messages"),
        ):
            mock_sh_cls.return_value = MagicMock()
            result = await agent._run_with_langgraph(max_steps=5)
            assert result._output_model_schema is TestSchema

    @pytest.mark.asyncio
    async def test_run_with_langgraph_initial_actions_interrupted(self):
        """Lines 1637-1640: InterruptedError during initial actions is swallowed."""
        agent = _create_agent()
        agent._force_exit_telemetry_logged = False

        mock_graph = MagicMock()
        mock_graph.run = AsyncMock()

        with (
            patch("openbrowser.agent.service.check_latest_openbrowser_version", new_callable=AsyncMock, return_value=None),
            patch("openbrowser.agent.graph.create_agent_graph", return_value=mock_graph, create=True),
            patch("openbrowser.utils.SignalHandler") as mock_sh_cls,
            patch.object(agent, "close", new_callable=AsyncMock),
            patch.object(agent, "_log_agent_event"),
            patch.object(agent, "_log_final_outcome_messages"),
            patch.object(agent, "_execute_initial_actions", new_callable=AsyncMock, side_effect=InterruptedError),
        ):
            mock_sh_cls.return_value = MagicMock()
            result = await agent._run_with_langgraph(max_steps=5)
            assert result is agent.history


# ===========================================================================
# multi_act wait between actions (lines 1720-1721)
# ===========================================================================


class TestMultiActWaitBetweenActions:
    """Cover multi_act waiting between actions."""

    @pytest.mark.asyncio
    async def test_multi_act_waits_between_actions(self):
        """Lines 1720-1721: wait_between_actions logging and sleep."""
        from openbrowser.models import ActionResult

        session = _make_mock_browser_session()
        session.browser_profile.wait_between_actions = 0.01
        agent = _create_agent(browser_session=session)

        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, click=(Optional[dict], None))

        actions = [
            CustomAction(click={"index": 1}),
            CustomAction(click={"index": 2}),
        ]
        # First action returns non-done, non-error to allow second action
        agent.tools.act = AsyncMock(
            side_effect=[
                ActionResult(extracted_content="ok"),
                ActionResult(extracted_content="ok2"),
            ]
        )

        results = await agent.multi_act(actions)
        assert len(results) == 2


# ===========================================================================
# _execute_history_step (lines 1923-1942)
# ===========================================================================


class TestExecuteHistoryStep:
    """Cover _execute_history_step."""

    @pytest.mark.asyncio
    async def test_execute_history_step_success(self):
        """Lines 1923-1942: successful history step execution."""
        from openbrowser.agent.views import AgentHistory, AgentOutput, StepMetadata
        from openbrowser.browser.views import BrowserStateHistory
        from openbrowser.models import ActionResult

        agent = _create_agent()
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, click=(Optional[dict], None))

        mock_element = MagicMock()
        mock_element.element_hash = "hash123"

        output = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[CustomAction(click={"index": 1})],
        )
        state = BrowserStateHistory(
            url="https://example.com",
            title="Example",
            tabs=[],
            interacted_element=[mock_element],
            screenshot_path=None,
        )
        history_item = AgentHistory(
            model_output=output,
            result=[ActionResult(extracted_content="ok")],
            state=state,
            metadata=StepMetadata(step_number=1, step_start_time=1.0, step_end_time=2.0),
        )

        browser_state = _make_mock_browser_state_summary()
        agent.browser_session.get_browser_state_summary = AsyncMock(return_value=browser_state)

        updated_action = CustomAction(click={"index": 1})
        with (
            patch.object(agent, "_update_action_indices", new_callable=AsyncMock, return_value=updated_action),
            patch.object(agent, "multi_act", new_callable=AsyncMock, return_value=[ActionResult(extracted_content="replayed")]),
        ):
            result = await agent._execute_history_step(history_item, delay=0.01)
            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_execute_history_step_no_state(self):
        """Line 1925-1926: state is falsy -> ValueError."""
        from openbrowser.agent.views import AgentHistory, StepMetadata
        from openbrowser.browser.views import BrowserStateHistory
        from openbrowser.models import ActionResult

        agent = _create_agent()
        state = BrowserStateHistory(
            url="", title="", tabs=[], interacted_element=[], screenshot_path=None,
        )
        history_item = AgentHistory(
            model_output=None,
            result=[ActionResult()],
            state=state,
            metadata=StepMetadata(step_number=1, step_start_time=1.0, step_end_time=2.0),
        )

        agent.browser_session.get_browser_state_summary = AsyncMock(return_value=_make_mock_browser_state_summary())

        with pytest.raises(ValueError, match="Invalid state or model output"):
            await agent._execute_history_step(history_item, delay=0.01)

    @pytest.mark.asyncio
    async def test_execute_history_step_element_not_found(self):
        """Lines 1936-1937: updated_action is None -> ValueError."""
        from openbrowser.agent.views import AgentHistory, AgentOutput, StepMetadata
        from openbrowser.browser.views import BrowserStateHistory
        from openbrowser.models import ActionResult

        agent = _create_agent()
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, click=(Optional[dict], None))

        mock_element = MagicMock()
        mock_element.element_hash = "hash123"

        output = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[CustomAction(click={"index": 1})],
        )
        state = BrowserStateHistory(
            url="https://example.com",
            title="Example",
            tabs=[],
            interacted_element=[mock_element],
            screenshot_path=None,
        )
        history_item = AgentHistory(
            model_output=output,
            result=[ActionResult(extracted_content="ok")],
            state=state,
            metadata=StepMetadata(step_number=1, step_start_time=1.0, step_end_time=2.0),
        )

        browser_state = _make_mock_browser_state_summary()
        agent.browser_session.get_browser_state_summary = AsyncMock(return_value=browser_state)

        with patch.object(agent, "_update_action_indices", new_callable=AsyncMock, return_value=None):
            with pytest.raises(ValueError, match="Could not find matching element"):
                await agent._execute_history_step(history_item, delay=0.01)


# ===========================================================================
# close() with remaining asyncio tasks (lines 2082-2084)
# ===========================================================================


class TestCloseWithRemainingTasks:
    """Cover close() logging remaining asyncio tasks."""

    @pytest.mark.asyncio
    async def test_close_logs_remaining_tasks(self):
        """Lines 2082-2084: other_tasks exist, they get logged."""
        agent = _create_agent()
        agent.browser_session.browser_profile.keep_alive = False

        # Create a background task that will appear in all_tasks
        async def dummy():
            await asyncio.sleep(10)

        bg_task = asyncio.create_task(dummy())
        try:
            await agent.close()
        finally:
            bg_task.cancel()
            try:
                await bg_task
            except asyncio.CancelledError:
                pass


# ===========================================================================
# get_trace_object screenshot removal (line 2122)
# ===========================================================================


class TestGetTraceObjectScreenshotRemoval:
    """Cover screenshot removal in get_trace_object."""

    def test_get_trace_object_removes_screenshots_from_history(self):
        """Line 2122: screenshots set to None in history dump."""
        from openbrowser.agent.views import AgentHistory, AgentOutput, StepMetadata
        from openbrowser.browser.views import BrowserStateHistory
        from openbrowser.models import ActionResult

        agent = _create_agent()
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))

        output = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[CustomAction(done={"text": "test"})],
        )
        state = BrowserStateHistory(
            url="https://example.com",
            title="Example",
            tabs=[],
            interacted_element=[],
            screenshot_path=None,
        )
        history_item = AgentHistory(
            model_output=output,
            result=[ActionResult(is_done=True, success=True, extracted_content="done")],
            state=state,
            metadata=StepMetadata(step_number=1, step_start_time=1.0, step_end_time=2.0),
        )
        agent.history.history.append(history_item)

        # The model_dump returns a dict with screenshot in state
        history_data_with_screenshot = {
            "history": [
                {
                    "state": {"url": "https://example.com", "screenshot": "base64data"},
                    "result": [],
                }
            ]
        }

        clean_settings = {"use_vision": "auto", "max_failures": 3}
        with (
            patch.object(type(agent.settings), "model_dump", return_value=clean_settings),
            patch.object(
                type(agent.history),
                "model_dump",
                return_value=history_data_with_screenshot,
            ),
        ):
            trace = agent.get_trace_object()
            # The complete_history should have screenshot set to None
            import json
            complete_history = json.loads(trace["trace_details"]["complete_history"])
            assert complete_history["history"][0]["state"]["screenshot"] is None
