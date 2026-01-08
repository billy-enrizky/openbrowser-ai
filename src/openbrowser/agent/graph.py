"""LangGraph-based Agent implementation for browser automation.

This module implements the agent workflow using LangGraph's StateGraph for
structured execution flow with proper state management.

The workflow uses node fusion for performance optimization:
    START -> step -> [continue? -> step : END]

Where each 'step' node performs: perceive -> plan -> execute -> finalize
This reduces LangGraph overhead by 4x compared to separate nodes.
"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

if TYPE_CHECKING:
    from openbrowser.agent.service import Agent
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from openbrowser.agent.views import (
    ActionResult,
    AgentHistory,
    AgentHistoryList,
    AgentOutput,
    AgentState,
    AgentStepInfo,
    BrowserStateHistory,
    StepMetadata,
)
from openbrowser.browser.views import BrowserStateSummary
from openbrowser.llm.base import BaseChatModel
from openbrowser.llm.messages import BaseMessage

logger = logging.getLogger(__name__)


class GraphState(TypedDict, total=False):
    """State for the LangGraph agent workflow.
    
    This state is passed between nodes in the graph and accumulates
    information as the agent progresses through steps.
    """
    # Core state
    messages: Annotated[list[BaseMessage], add_messages]
    
    # Browser state
    browser_state_summary: BrowserStateSummary | None
    screenshot: str | None  # Base64-encoded screenshot
    
    # Agent state
    step_number: int
    max_steps: int
    model_output: AgentOutput | None
    action_results: list[ActionResult]
    
    # Control flow
    is_done: bool
    consecutive_failures: int
    error: str | None
    
    # Timing
    step_start_time: float


class AgentGraphBuilder:
    """Builds a LangGraph StateGraph for the browser automation agent.
    
    This class constructs a graph with the following nodes:
    - perceive: Captures browser state (screenshot + DOM)
    - plan: LLM decides next actions based on state
    - execute: Runs the planned actions
    
    The graph uses conditional edges to determine when to continue
    or terminate based on task completion or failure conditions.
    """
    
    def __init__(
        self,
        agent: 'Agent',  # Forward reference to avoid circular import
    ):
        """Initialize the graph builder.
        
        Args:
            agent: The Agent instance that provides browser session,
                   LLM, tools, and other components needed for execution.
        """
        self.agent = agent
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph StateGraph for agent execution.
        
        The graph uses node fusion for performance optimization:
        1. step: Combined perceive + plan + execute + finalize (single node per step)
        2. check: Decide whether to continue or end
        
        This reduces LangGraph overhead by minimizing state transitions
        while maintaining the same behavior as browser_use's traditional loop.
        
        Returns:
            Compiled StateGraph ready for execution.
        """
        # Create the graph with our state schema
        graph = StateGraph(GraphState)
        
        # Single fused node for entire step (perceive -> plan -> execute -> finalize)
        graph.add_node("step", self._step_node)
        
        # Simple flow: START -> step -> [continue? -> step : END]
        graph.add_edge(START, "step")
        
        # Conditional edge from step to decide next action
        graph.add_conditional_edges(
            "step",
            self._should_continue,
            {
                "continue": "step",   # Continue to next step
                "done": END,          # Task completed
                "error": END,         # Too many failures
            }
        )
        
        return graph.compile()
    
    async def _step_node(self, state: GraphState) -> GraphState:
        """Combined step node: perceive + plan + execute + finalize in one.
        
        This fused node performs the entire step cycle:
        1. Perceive: Capture browser state (screenshot + DOM)
        2. Plan: LLM decides next actions
        3. Execute: Run the planned actions
        4. Finalize: Record history and update state
        
        Node fusion reduces LangGraph overhead by minimizing state transitions.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with step results
        """
        step_start_time = time.time()
        step_number = state.get("step_number", 0)
        max_steps = state.get("max_steps", 100)
        
        # Log step info like browser_use does
        self.agent.logger.info('\n')
        self.agent.logger.info(f'Step {step_number}:')
        
        browser_state_summary = None
        model_output = None
        action_results = []
        is_done = False
        consecutive_failures = state.get("consecutive_failures", 0)
        error = None
        
        try:
            # ===== PERCEIVE PHASE =====
            await self.agent._check_stop_or_pause()
            
            browser_state_summary = await self.agent.browser_session.get_browser_state_summary(
                include_screenshot=True,
                include_recent_events=self.agent.include_recent_events,
            )
            
            await self.agent._check_and_update_downloads(f'Step {step_number}: after getting browser state')
            
            url = browser_state_summary.url if browser_state_summary else ''
            url_short = url[:50] + '...' if len(url) > 50 else url
            interactive_count = len(browser_state_summary.dom_state.selector_map) if browser_state_summary else 0
            self.agent.logger.debug(f'Evaluating page with {interactive_count} interactive elements on: {url_short}')
            
            await self.agent._update_action_models_for_page(browser_state_summary.url)
            
            page_filtered_actions = self.agent.tools.registry.get_prompt_description(
                browser_state_summary.url
            )
            
            step_info = AgentStepInfo(step_number=step_number, max_steps=max_steps)
            
            self.agent._message_manager.create_state_messages(
                browser_state_summary=browser_state_summary,
                model_output=self.agent.state.last_model_output,
                result=self.agent.state.last_result,
                step_info=step_info,
                use_vision=self.agent.settings.use_vision,
                page_filtered_actions=page_filtered_actions,
                sensitive_data=self.agent.sensitive_data,
                available_file_paths=self.agent.available_file_paths,
            )
            
            await self.agent._force_done_after_last_step(step_info)
            await self.agent._force_done_after_failure()
            
            # ===== PLAN PHASE =====
            await self.agent._check_stop_or_pause()
            
            input_messages = self.agent._message_manager.get_messages()
            
            model_output = await asyncio.wait_for(
                self.agent._get_model_output_with_retry(input_messages),
                timeout=self.agent.settings.llm_timeout
            )
            
            self.agent.state.last_model_output = model_output
            
            await self.agent._check_stop_or_pause()
            
            if browser_state_summary:
                await self.agent._handle_post_llm_processing(browser_state_summary, input_messages)
            
            await self.agent._check_stop_or_pause()
            
            # ===== EXECUTE PHASE =====
            if model_output and model_output.action:
                self.agent.logger.debug(f'Step {step_number}: Executing {len(model_output.action)} actions...')
                
                await self.agent._check_stop_or_pause()
                
                action_results = await self.agent.multi_act(model_output.action)
                
                self.agent.state.last_result = action_results
                
                await self.agent._check_and_update_downloads('after executing actions')
                
                is_done = any(r.is_done for r in action_results if r)
                
                has_error = any(r.error for r in action_results if r)
                if has_error and len(action_results) == 1:
                    consecutive_failures += 1
                    self.agent.logger.debug(f'Step {step_number}: Consecutive failures: {consecutive_failures}')
                elif consecutive_failures > 0:
                    consecutive_failures = 0
                    self.agent.logger.debug(f'Step {step_number}: Consecutive failures reset to: {consecutive_failures}')
                
                self.agent.state.consecutive_failures = consecutive_failures
                
                if action_results and len(action_results) > 0 and action_results[-1].is_done:
                    self.agent.logger.info(f'\n Final Result:\n{action_results[-1].extracted_content}\n\n')
                    if action_results[-1].attachments:
                        total_attachments = len(action_results[-1].attachments)
                        for i, file_path in enumerate(action_results[-1].attachments):
                            self.agent.logger.info(f'Attachment {i + 1 if total_attachments > 1 else ""}: {file_path}')
            else:
                error = "No actions to execute"
                consecutive_failures += 1
            
            # ===== FINALIZE PHASE =====
            step_end_time = time.time()
            
            if action_results and browser_state_summary:
                metadata = StepMetadata(
                    step_number=step_number,
                    step_start_time=step_start_time,
                    step_end_time=step_end_time,
                )
                
                await self.agent._make_history_item(
                    model_output,
                    browser_state_summary,
                    action_results,
                    metadata,
                    state_message=self.agent._message_manager.last_state_message_text,
                )
                
                step_duration = step_end_time - step_start_time
                action_count = len(action_results)
                self.agent.logger.debug(
                    f'Step {step_number}: Ran {action_count} action{"" if action_count == 1 else "s"} in {step_duration:.2f}s'
                )
            
            self.agent.save_file_system_state()
            
            if browser_state_summary and model_output:
                try:
                    from openbrowser.agent.cloud_events import CreateAgentStepEvent
                    
                    actions_data = []
                    if model_output.action:
                        for action in model_output.action:
                            action_dict = action.model_dump() if hasattr(action, 'model_dump') else {}
                            actions_data.append(action_dict)
                    
                    step_event = CreateAgentStepEvent.from_agent_step(
                        self.agent,
                        model_output,
                        action_results,
                        actions_data,
                        browser_state_summary,
                    )
                    self.agent.eventbus.dispatch(step_event)
                except Exception as e:
                    self.agent.logger.debug(f'Failed to emit step event: {e}')
            
            self.agent.state.n_steps = step_number + 1
            
            return {
                "step_number": step_number + 1,
                "browser_state_summary": browser_state_summary,
                "screenshot": browser_state_summary.screenshot if browser_state_summary else None,
                "model_output": model_output,
                "action_results": action_results,
                "is_done": is_done,
                "consecutive_failures": consecutive_failures,
                "error": error,
            }
            
        except InterruptedError:
            self.agent.logger.info('Agent interrupted')
            return {
                "step_number": step_number + 1,
                "is_done": True,
                "error": "Agent interrupted",
            }
        except asyncio.TimeoutError:
            error_msg = f"LLM call timed out after {self.agent.settings.llm_timeout} seconds"
            self.agent.logger.error(error_msg)
            return {
                "step_number": step_number + 1,
                "error": error_msg,
                "consecutive_failures": consecutive_failures + 1,
            }
        except Exception as e:
            logger.error(f"Error in step node: {e}")
            self.agent.state.last_result = [ActionResult(error=str(e))]
            return {
                "step_number": step_number + 1,
                "action_results": [ActionResult(error=str(e))],
                "error": str(e),
                "consecutive_failures": consecutive_failures + 1,
            }
    
    def _should_continue(self, state: GraphState) -> Literal["continue", "done", "error"]:
        """Determine if the agent should continue, finish, or handle error.
        
        This is called AFTER the step node, so history has already been recorded.
        
        Args:
            state: Current graph state
            
        Returns:
            "continue" to keep going, "done" if task complete, "error" if failed
        """
        # Check for done condition (task completed)
        if state.get("is_done"):
            self.agent.logger.info('Task completed successfully')
            return "done"
        
        # Check for agent stopped state
        if self.agent.state.stopped:
            self.agent.logger.info('Agent stopped')
            return "done"
        
        # Check for agent paused state
        if self.agent.state.paused:
            self.agent.logger.info('Agent paused')
            return "done"
        
        # Check for max steps
        step_number = state.get("step_number", 0)
        max_steps = state.get("max_steps", 100)
        if step_number >= max_steps:
            self.agent.logger.info(f'Reached max steps ({max_steps})')
            return "done"
        
        # Check for too many failures (like browser_use)
        consecutive_failures = state.get("consecutive_failures", 0)
        max_failures = self.agent.settings.max_failures
        # Account for final_response_after_failure setting
        effective_max = max_failures + int(self.agent.settings.final_response_after_failure)
        if consecutive_failures >= effective_max:
            self.agent.logger.error(f'Stopping due to {max_failures} consecutive failures')
            return "error"
        
        return "continue"
    
    async def run(
        self,
        max_steps: int = 100,
        initial_state: GraphState | None = None,
    ) -> AgentHistoryList:
        """Run the agent graph.
        
        Args:
            max_steps: Maximum number of steps to execute
            initial_state: Optional initial state to start with
            
        Returns:
            AgentHistoryList with execution history
        """
        import inspect
        
        # Initialize state
        state: GraphState = initial_state or {
            "messages": [],
            "browser_state_summary": None,
            "screenshot": None,
            "step_number": 0,
            "max_steps": max_steps,
            "model_output": None,
            "action_results": [],
            "is_done": False,
            "consecutive_failures": 0,
            "error": None,
            "step_start_time": time.time(),
        }
        
        # Calculate recursion limit: 1 fused node per step (node fusion optimization)
        # Add buffer for safety
        nodes_per_step = 1
        recursion_limit = (max_steps * nodes_per_step) + 10
        
        # Run the graph with ainvoke (faster than astream when we don't need streaming)
        config = {"recursion_limit": recursion_limit}
        final_state = await self.graph.ainvoke(state, config=config)
        
        # Call done callback if task is done (like browser_use)
        if self.agent.history.is_done():
            await self.agent.log_completion()
            if self.agent.register_done_callback:
                if inspect.iscoroutinefunction(self.agent.register_done_callback):
                    await self.agent.register_done_callback(self.agent.history)
                else:
                    self.agent.register_done_callback(self.agent.history)
        
        return self.agent.history


def create_agent_graph(agent: 'Agent') -> AgentGraphBuilder:
    """Factory function to create an agent graph.
    
    Args:
        agent: The Agent instance to build the graph for
        
    Returns:
        AgentGraphBuilder instance with compiled graph
    """
    return AgentGraphBuilder(agent)


