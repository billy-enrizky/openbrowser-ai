"""LangGraph workflow for browser automation agent following browser-use pattern."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from src.openbrowser.agent.message_manager import MessageManager, MessageManagerState
from src.openbrowser.agent.prompts import SystemPrompt
from src.openbrowser.agent.views import (
    ActionResult,
    AgentHistory,
    AgentHistoryList,
    AgentOutput,
    AgentSettings,
    AgentStepInfo,
    BrowserStateHistory,
    StepMetadata,
)
from src.openbrowser.browser.dom import DomService
from src.openbrowser.browser.profile import BrowserProfile
from src.openbrowser.browser.session import BrowserSession
from src.openbrowser.tools.actions import Tools

from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the browser automation agent."""
    
    messages: Annotated[list[BaseMessage], add_messages]
    screenshot: str
    dom_tree: str
    url: str
    root_goal: str
    n_steps: int
    max_steps: int
    consecutive_failures: int
    model_output: AgentOutput | None
    last_result: list[ActionResult] | None
    is_done: bool


class BrowserAgent:
    """Browser automation agent using LangGraph workflow with browser-use patterns."""

    def __init__(
        self,
        headless: bool = True,
        model_name: str = "gpt-4o",
        llm_provider: str = "openai",
        api_key: str | None = None,
        browser_profile: BrowserProfile | None = None,
        use_vision: bool | Literal['auto'] = 'auto',
        max_failures: int = 3,
        max_actions_per_step: int = 4,
        use_thinking: bool = True,
        flash_mode: bool = False,
        max_steps: int = 50,
        max_history_items: int | None = None,
        override_system_message: str | None = None,
        extend_system_message: str | None = None,
        # Callback support
        register_new_step_callback: Any | None = None,
        register_done_callback: Any | None = None,
        register_should_stop_callback: Any | None = None,
    ):
        """Initialize BrowserAgent."""
        logger.info(f"Initializing BrowserAgent with provider: {llm_provider}, model: {model_name}")

        if browser_profile is None:
            browser_profile = BrowserProfile(headless=headless)

        self.browser_session = BrowserSession(browser_profile=browser_profile)
        self.tools = Tools(self.browser_session)

        if llm_provider == "google":
            from src.openbrowser.llm.google import ChatGoogle
            if api_key:
                self.llm = ChatGoogle(model=model_name, temperature=0, api_key=api_key)
            else:
                self.llm = ChatGoogle(model=model_name, temperature=0)
        else:
            from src.openbrowser.llm.openai import ChatOpenAI
            self.llm = ChatOpenAI(model=model_name, temperature=0, api_key=api_key)

        self.settings = AgentSettings(
            use_vision=use_vision,
            max_failures=max_failures,
            max_actions_per_step=max_actions_per_step,
            use_thinking=use_thinking,
            flash_mode=flash_mode,
            max_history_items=max_history_items,
        )
        self.max_steps = max_steps

        self.system_prompt = SystemPrompt(
            max_actions_per_step=max_actions_per_step,
            override_system_message=override_system_message,
            extend_system_message=extend_system_message,
            use_thinking=use_thinking,
            flash_mode=flash_mode,
        )

        self.message_manager: MessageManager | None = None
        self.history = AgentHistoryList()
        
        # Callback support
        self.register_new_step_callback = register_new_step_callback
        self.register_done_callback = register_done_callback
        self.register_should_stop_callback = register_should_stop_callback
        
        self.memory = MemorySaver()
        self.app = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("perceive", self.perceive_node)
        workflow.add_node("step", self.step_node)
        workflow.add_node("execute", self.execute_node)

        workflow.set_entry_point("perceive")
        
        workflow.add_edge("perceive", "step")
        workflow.add_conditional_edges(
            "step",
            self._route_step,
            {
                "execute": "execute",
                "end": END,
            },
        )
        workflow.add_edge("execute", "perceive")

        return workflow.compile(checkpointer=self.memory)

    async def perceive_node(self, state: AgentState) -> dict:
        """Perceive current browser state - capture screenshot and DOM."""
        logger.info(f"Perceive: Step {state.get('n_steps', 0) + 1}")
        
        if not self.browser_session.agent_focus:
            raise RuntimeError("Browser not started. Call browser_session.start() first.")

        cdp_session = self.browser_session.agent_focus
        client = cdp_session.cdp_client
        session_id = cdp_session.session_id

        try:
            await client.send.Page.enable(session_id=session_id)
        except Exception:
            pass

        await asyncio.sleep(0.5)

        screenshot_result = await client.send.Page.captureScreenshot(
            params={"format": "png"}, session_id=session_id
        )
        screenshot = screenshot_result.get("data", "")

        dom_state = await DomService.get_clickable_elements(client, session_id)
        self.tools.update_state(dom_state)

        current_url = ""
        try:
            nav_history = await client.send.Page.getNavigationHistory(session_id=session_id)
            idx = nav_history.get("currentIndex", 0)
            entries = nav_history.get("entries", [])
            if entries and idx < len(entries):
                current_url = entries[idx].get("url", "")
        except Exception:
            pass

        return {
            "screenshot": screenshot,
            "dom_tree": dom_state.element_tree,
            "url": current_url,
        }

    async def step_node(self, state: AgentState) -> dict:
        """Single LLM call with unified AgentOutput."""
        step_number = state.get("n_steps", 0)
        
        logger.info(f"Step {step_number + 1}/{self.max_steps}")

        if self.message_manager is None:
            raise RuntimeError("Message manager not initialized")

        from src.openbrowser.browser.dom import DomState
        dom_state = DomState(
            element_tree=state.get("dom_tree", ""),
            selector_map=self.tools._selector_map,
        )

        step_info = AgentStepInfo(step_number=step_number, max_steps=self.max_steps)
        action_descriptions = self.tools.get_prompt_description()

        self.message_manager.create_state_messages(
            dom_state=dom_state,
            url=state.get("url", ""),
            screenshot=state.get("screenshot"),
            model_output=state.get("model_output"),
            result=state.get("last_result"),
            step_info=step_info,
            use_vision=self.settings.use_vision,
            action_descriptions=action_descriptions,
        )

        messages = self.message_manager.get_messages()
        action_model = self.tools.create_action_model()
        
        if self.settings.flash_mode:
            output_model = AgentOutput.type_with_custom_actions_flash_mode(action_model)
        else:
            output_model = AgentOutput.type_with_custom_actions(action_model)

        try:
            llm_with_output = self.llm.with_structured_output(output_model, method="function_calling")
            result: AgentOutput = await llm_with_output.ainvoke(messages)
            
            if result is None:
                raise ValueError("LLM returned None")

            logger.info(f"  Memory: {result.memory}")
            logger.info(f"  Next goal: {result.next_goal}")
            logger.info(f"  Actions: {len(result.action)}")

            return {
                "model_output": result,
                "n_steps": step_number + 1,
                "consecutive_failures": 0,
            }

        except Exception as e:
            logger.error(f"Step failed: {e}")
            consecutive_failures = state.get("consecutive_failures", 0) + 1
            
            if consecutive_failures >= self.settings.max_failures:
                logger.error(f"Max failures ({self.settings.max_failures}) reached")
                return {
                    "is_done": True,
                    "n_steps": step_number + 1,
                    "consecutive_failures": consecutive_failures,
                    "model_output": None,
                }
            
            return {
                "n_steps": step_number + 1,
                "consecutive_failures": consecutive_failures,
                "model_output": None,
            }

    async def execute_node(self, state: AgentState) -> dict:
        """Execute actions from the model output."""
        logger.info("Executing actions")
        
        model_output = state.get("model_output")
        if not model_output or not model_output.action:
            return {"last_result": []}

        results: list[ActionResult] = []
        is_done = False

        for action in model_output.action:
            action_dict = action.model_dump(exclude_none=True)
            
            if not action_dict:
                continue

            action_name = next(iter(action_dict.keys()))
            action_params = action_dict[action_name]
            
            if action_params is None:
                action_params = {}

            logger.info(f"  Executing: {action_name}({action_params})")

            try:
                result = await self.tools.execute_action(action_name, action_params)
                results.append(result)
                
                if result.is_done:
                    is_done = True
                    break
                    
                if result.error:
                    logger.warning(f"  Action error: {result.error}")

            except Exception as e:
                logger.error(f"  Action failed: {e}")
                results.append(ActionResult(error=str(e)))

        step_end_time = time.time()
        step_number = state.get("n_steps", 1) - 1
        
        browser_state = BrowserStateHistory(
            url=state.get("url"),
            screenshot=state.get("screenshot"),
        )
        
        history_item = AgentHistory(
            model_output=model_output,
            result=results,
            state=browser_state,
            metadata=StepMetadata(
                step_start_time=step_end_time - 1,
                step_end_time=step_end_time,
                step_number=step_number,
            ),
        )
        self.history.add_item(history_item)

        # Invoke new_step_callback if registered
        if self.register_new_step_callback:
            try:
                await self.register_new_step_callback(browser_state, model_output, step_number)
            except Exception as e:
                logger.warning(f"new_step_callback failed: {e}")

        return {
            "last_result": results,
            "is_done": is_done,
        }

    def _route_step(self, state: AgentState) -> Literal["execute", "end"]:
        """Route after step node."""
        if state.get("is_done"):
            logger.info("Task completed")
            return "end"

        if state.get("n_steps", 0) >= self.max_steps:
            logger.info("Max steps reached")
            return "end"

        model_output = state.get("model_output")
        if not model_output or not model_output.action:
            if state.get("consecutive_failures", 0) >= self.settings.max_failures:
                return "end"
            return "execute"

        for action in model_output.action:
            action_dict = action.model_dump(exclude_none=True)
            if "done" in action_dict:
                return "end"

        return "execute"

    async def _check_should_stop(self) -> bool:
        """Check if the agent should stop via the callback."""
        if self.register_should_stop_callback:
            try:
                return await self.register_should_stop_callback()
            except Exception as e:
                logger.warning(f"should_stop_callback failed: {e}")
        return False

    async def run(self, goal: str, max_steps: int | None = None) -> AgentHistoryList:
        """Run the agent to complete a goal."""
        logger.info(f"Starting agent run: {goal}")
        
        if max_steps:
            self.max_steps = max_steps

        self.message_manager = MessageManager(
            task=goal,
            system_message=self.system_prompt.get_system_message(),
            max_history_items=self.settings.max_history_items,
        )

        self.history = AgentHistoryList()

        try:
            await self.browser_session.start()

            config = {"configurable": {"thread_id": "1"}}
            initial_state: AgentState = {
                "messages": [HumanMessage(content=goal)],
                "screenshot": "",
                "dom_tree": "",
                "url": "",
                "root_goal": goal,
                "n_steps": 0,
                "max_steps": self.max_steps,
                "consecutive_failures": 0,
                "model_output": None,
                "last_result": None,
                "is_done": False,
            }

            await self.app.ainvoke(
                initial_state, 
                config={**config, "recursion_limit": self.max_steps * 3}
            )

            # Invoke done_callback if registered
            if self.register_done_callback:
                try:
                    await self.register_done_callback(self.history)
                except Exception as e:
                    logger.warning(f"done_callback failed: {e}")

            return self.history

        finally:
            try:
                await self.browser_session.stop()
            except Exception as e:
                logger.warning(f"Error stopping browser session: {e}")

    async def run_step(self, state: AgentState) -> AgentState:
        """Run a single step (for external control)."""
        result = await self.app.ainvoke(state, config={"configurable": {"thread_id": "1"}})
        return result

    async def rerun_history(
        self,
        history: AgentHistoryList,
        max_retries: int = 3,
        skip_failures: bool = True,
        delay_between_actions: float = 2.0,
    ) -> list[ActionResult]:
        """
        Rerun a saved history of actions with error handling and retry logic.

        Args:
            history: The history to replay
            max_retries: Maximum number of retries per action
            skip_failures: Whether to skip failed actions or stop execution
            delay_between_actions: Delay between actions in seconds

        Returns:
            List of action results
        """
        # Initialize browser session
        await self.browser_session.start()

        results: list[ActionResult] = []

        for i, history_item in enumerate(history.history):
            goal = history_item.model_output.next_goal if history_item.model_output else ''
            step_num = history_item.metadata.step_number if history_item.metadata else i
            step_name = 'Initial actions' if step_num == 0 else f'Step {step_num}'
            logger.info(f'Replaying {step_name} ({i + 1}/{len(history.history)}): {goal}')

            if (
                not history_item.model_output
                or not history_item.model_output.action
                or history_item.model_output.action == [None]
            ):
                logger.warning(f'{step_name}: No action to replay, skipping')
                results.append(ActionResult(error='No action to replay'))
                continue

            retry_count = 0
            while retry_count < max_retries:
                try:
                    result = await self._execute_history_step(history_item, delay_between_actions)
                    results.extend(result)
                    break

                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        error_msg = f'{step_name} failed after {max_retries} attempts: {str(e)}'
                        logger.error(error_msg)
                        if not skip_failures:
                            results.append(ActionResult(error=error_msg))
                            raise RuntimeError(error_msg)
                    else:
                        logger.warning(f'{step_name} failed (attempt {retry_count}/{max_retries}), retrying...')
                        await asyncio.sleep(delay_between_actions)

        await self.browser_session.stop()
        return results

    async def _execute_history_step(
        self,
        history_item: AgentHistory,
        delay: float
    ) -> list[ActionResult]:
        """Execute a single step from history."""
        if not history_item.model_output:
            raise ValueError('Invalid model output')

        results: list[ActionResult] = []

        for action in history_item.model_output.action:
            action_dict = action.model_dump(exclude_none=True)

            if not action_dict:
                continue

            action_name = next(iter(action_dict.keys()))
            action_params = action_dict[action_name]

            if action_params is None:
                action_params = {}

            logger.info(f'  Replaying: {action_name}({action_params})')

            try:
                result = await self.tools.execute_action(action_name, action_params)
                results.append(result)

                if result.is_done:
                    break

            except Exception as e:
                logger.error(f'  Action failed: {e}')
                raise

        await asyncio.sleep(delay)
        return results

    async def load_and_rerun(
        self,
        history_file: str | Path | None = None,
        **kwargs
    ) -> list[ActionResult]:
        """
        Load history from file and rerun it.

        Args:
            history_file: Path to the history file
            **kwargs: Additional arguments passed to rerun_history
        """
        if not history_file:
            history_file = 'AgentHistory.json'
        history = AgentHistoryList.load_from_file(history_file)
        return await self.rerun_history(history, **kwargs)

    def save_history(self, file_path: str | Path | None = None) -> None:
        """Save the history to a file."""
        if not file_path:
            file_path = 'AgentHistory.json'
        self.history.save_to_file(file_path)
