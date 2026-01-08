"""LangGraph-based Agent implementation for browser automation.

This module implements the agent workflow using LangGraph's StateGraph for
structured execution flow with proper state management.

Performance optimizations:
- Node fusion (4 nodes -> 1)
- Minimal state (5 control fields only)
- ainvoke (not astream)
- Parallel async operations where possible
- Skip unnecessary checks based on settings
- Lazy evaluation of expensive operations
"""

import asyncio
import inspect
import logging
import time
from typing import TYPE_CHECKING, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

if TYPE_CHECKING:
    from openbrowser.agent.service import Agent

from openbrowser.agent.views import (
    ActionResult,
    AgentHistoryList,
    AgentStepInfo,
    StepMetadata,
)

# Import at module level for performance
try:
    from openbrowser.agent.cloud_events import CreateAgentStepEvent
    _HAS_CLOUD_EVENTS = True
except ImportError:
    _HAS_CLOUD_EVENTS = False

logger = logging.getLogger(__name__)


class GraphState(TypedDict, total=False):
    """Minimal state for the LangGraph agent workflow.
    
    Only contains control flow fields - actual data is stored in agent.state
    to minimize state copying overhead between steps.
    """
    # Control flow (minimal state for performance)
    step_number: int
    max_steps: int
    is_done: bool
    consecutive_failures: int


class AgentGraphBuilder:
    """Builds a LangGraph StateGraph for the browser automation agent.
    
    Uses graph caching and minimal state for optimal performance.
    """
    __slots__ = ('agent', 'graph', '_has_downloads')
    
    def __init__(self, agent: 'Agent'):
        self.agent = agent
        self._has_downloads = agent.has_downloads_path  # Cache this check
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
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
        """Fused step: perceive -> plan -> execute -> finalize."""
        t0 = time.time()
        step = state.get("step_number", 0)
        max_steps = state.get("max_steps", 100)
        failures = state.get("consecutive_failures", 0)
        agent = self.agent
        
        agent.logger.info(f'\nStep {step}:')
        
        try:
            # Check stop/pause once
            await agent._check_stop_or_pause()
            
            # ===== PERCEIVE (with parallel download check) =====
            browser_state, _ = await asyncio.gather(
                agent.browser_session.get_browser_state_summary(
                    include_screenshot=True,
                    include_recent_events=agent.include_recent_events,
                ),
                agent._check_and_update_downloads(f'Step {step}') if self._has_downloads else asyncio.sleep(0),
            )
            
            # Update action models (skip if URL unchanged - optimization)
            url = browser_state.url if browser_state else ''
            await agent._update_action_models_for_page(url)
            
            step_info = AgentStepInfo(step_number=step, max_steps=max_steps)
            
            # Create messages
            agent._message_manager.create_state_messages(
                browser_state_summary=browser_state,
                model_output=agent.state.last_model_output,
                result=agent.state.last_result,
                step_info=step_info,
                use_vision=agent.settings.use_vision,
                page_filtered_actions=agent.tools.registry.get_prompt_description(url),
                sensitive_data=agent.sensitive_data,
                available_file_paths=agent.available_file_paths,
            )
            
            # Force done checks (run in parallel)
            await asyncio.gather(
                agent._force_done_after_last_step(step_info),
                agent._force_done_after_failure(),
            )
            
            # ===== PLAN =====
            model_output = await asyncio.wait_for(
                agent._get_model_output_with_retry(agent._message_manager.get_messages()),
                timeout=agent.settings.llm_timeout
            )
            agent.state.last_model_output = model_output
            
            # Post-LLM processing (can run while we check actions)
            if browser_state:
                await agent._handle_post_llm_processing(browser_state, agent._message_manager.get_messages())
            
            # ===== EXECUTE =====
            is_done = False
            results = []
            
            if model_output and model_output.action:
                results = await agent.multi_act(model_output.action)
                agent.state.last_result = results
                
                # Check downloads after actions (only if configured)
                if self._has_downloads:
                    await agent._check_and_update_downloads('after actions')
                
                is_done = any(r.is_done for r in results if r)
                
                # Update failures
                if any(r.error for r in results if r) and len(results) == 1:
                    failures += 1
                elif failures > 0:
                    failures = 0
                agent.state.consecutive_failures = failures
                
                # Log final result
                if results and results[-1].is_done:
                    agent.logger.info(f'\n Final Result:\n{results[-1].extracted_content}\n\n')
            else:
                failures += 1
            
            # ===== FINALIZE (parallel operations) =====
            t1 = time.time()
            
            # Run finalization tasks in parallel
            finalize_tasks = []
            
            if results and browser_state:
                finalize_tasks.append(agent._make_history_item(
                    model_output, browser_state, results,
                    StepMetadata(step_number=step, step_start_time=t0, step_end_time=t1),
                    state_message=agent._message_manager.last_state_message_text,
                ))
            
            # Cloud events (fire and forget style)
            if _HAS_CLOUD_EVENTS and browser_state and model_output:
                try:
                    actions_data = [a.model_dump() for a in (model_output.action or []) if hasattr(a, 'model_dump')]
                    agent.eventbus.dispatch(CreateAgentStepEvent.from_agent_step(
                        agent, model_output, results, actions_data, browser_state,
                    ))
                except Exception:
                    pass
            
            if finalize_tasks:
                await asyncio.gather(*finalize_tasks)
            
            # Sync operations (fast)
            agent.save_file_system_state()
            agent.state.n_steps = step + 1
            
            return {"step_number": step + 1, "is_done": is_done, "consecutive_failures": failures}
            
        except InterruptedError:
            agent.logger.info('Interrupted')
            return {"step_number": step + 1, "is_done": True}
        except asyncio.TimeoutError:
            agent.logger.error(f"LLM timeout")
            return {"step_number": step + 1, "consecutive_failures": failures + 1}
        except Exception as e:
            logger.error(f"Step error: {e}")
            agent.state.last_result = [ActionResult(error=str(e))]
            return {"step_number": step + 1, "consecutive_failures": failures + 1}
    
    def _should_continue(self, state: GraphState) -> Literal["continue", "done", "error"]:
        if state.get("is_done"):
            self.agent.logger.info('Task completed')
            return "done"
        if self.agent.state.stopped or self.agent.state.paused:
            return "done"
        if state.get("step_number", 0) >= state.get("max_steps", 100):
            return "done"
        max_fail = self.agent.settings.max_failures + int(self.agent.settings.final_response_after_failure)
        if state.get("consecutive_failures", 0) >= max_fail:
            return "error"
        return "continue"
    
    async def run(self, max_steps: int = 100) -> AgentHistoryList:
        state: GraphState = {"step_number": 0, "max_steps": max_steps, "is_done": False, "consecutive_failures": 0}
        await self.graph.ainvoke(state, config={"recursion_limit": max_steps + 10})
        
        # Done callback
        if self.agent.history.is_done():
            await self.agent.log_completion()
            cb = self.agent.register_done_callback
            if cb:
                await cb(self.agent.history) if inspect.iscoroutinefunction(cb) else cb(self.agent.history)
        
        return self.agent.history


def create_agent_graph(agent: 'Agent') -> AgentGraphBuilder:
    """Factory function to create an agent graph."""
    return AgentGraphBuilder(agent)