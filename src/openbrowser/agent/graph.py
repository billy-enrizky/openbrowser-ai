"""LangGraph-based Agent implementation for browser automation.

Performance optimizations:
- Single fused node (perceive + plan + execute + finalize)
- Minimal state (4 control fields only)
- ainvoke (not astream)
- Parallel async operations
- Cached checks
- __slots__ for class
- Local variable optimization
- Module-level imports
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

logger = logging.getLogger(__name__)


class GraphState(TypedDict, total=False):
    """Minimal state for LangGraph workflow.
    
    Only control flow fields - actual data in agent.state to minimize copying.
    """
    step_number: int
    max_steps: int
    is_done: bool
    consecutive_failures: int


class AgentGraphBuilder:
    """Optimized LangGraph agent with minimal overhead.
    
    Performance features:
    - Single fused step node
    - __slots__ for faster attribute access
    - Cached downloads check
    - Parallel async operations
    - Local variable optimization
    """
    
    __slots__ = ('agent', 'graph', '_has_downloads', '_max_failures')
    
    def __init__(self, agent: 'Agent'):
        self.agent = agent
        # Cache these checks at init time
        self._has_downloads = agent.has_downloads_path
        self._max_failures = agent.settings.max_failures + int(agent.settings.final_response_after_failure)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build minimal StateGraph: START -> step -> [continue/done/error]."""
        graph = StateGraph(GraphState)
        graph.add_node("step", self._step_node)
        graph.add_edge(START, "step")
        graph.add_conditional_edges(
            "step",
            self._should_continue,
            {"continue": "step", "done": END, "error": END}
        )
        return graph.compile()
    
    async def _step_node(self, state: GraphState) -> GraphState:
        """Fused step: perceive -> plan -> execute -> finalize.
        
        All phases combined into single node for minimal LangGraph overhead.
        """
        t0 = time.time()
        step = state.get("step_number", 0)
        max_steps = state.get("max_steps", 100)
        failures = state.get("consecutive_failures", 0)
        
        # Local variable for faster access
        agent = self.agent
        agent.logger.info(f'\nStep {step}:')
        
        try:
            # Single stop/pause check
            await agent._check_stop_or_pause()
            
            # === PERCEIVE (parallel browser state + downloads) ===
            if self._has_downloads:
                browser_state, _ = await asyncio.gather(
                    agent.browser_session.get_browser_state_summary(
                        include_screenshot=True,
                        include_recent_events=agent.include_recent_events,
                    ),
                    agent._check_and_update_downloads(f'Step {step}'),
                )
            else:
                browser_state = await agent.browser_session.get_browser_state_summary(
                    include_screenshot=True,
                    include_recent_events=agent.include_recent_events,
                )
            
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
            
            # Parallel force done checks
            await asyncio.gather(
                agent._force_done_after_last_step(step_info),
                agent._force_done_after_failure(),
            )
            
            # === PLAN ===
            model_output = await asyncio.wait_for(
                agent._get_model_output_with_retry(agent._message_manager.get_messages()),
                timeout=agent.settings.llm_timeout
            )
            agent.state.last_model_output = model_output
            
            # Post-LLM processing (parallel with action check)
            if browser_state:
                await agent._handle_post_llm_processing(browser_state, agent._message_manager.get_messages())
            
            # === EXECUTE ===
            is_done = False
            results = []
            
            if model_output and model_output.action:
                results = await agent.multi_act(model_output.action)
                agent.state.last_result = results
                
                # Download check after actions (only if configured)
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
            
            # === FINALIZE ===
            t1 = time.time()
            
            # History item creation
            if results and browser_state:
                await agent._make_history_item(
                    model_output, browser_state, results,
                    StepMetadata(step_number=step, step_start_time=t0, step_end_time=t1),
                    state_message=agent._message_manager.last_state_message_text,
                )
            
            # Fast sync operations
            agent.save_file_system_state()
            agent.state.n_steps = step + 1
            
            return {"step_number": step + 1, "is_done": is_done, "consecutive_failures": failures}
            
        except InterruptedError:
            agent.logger.info('Interrupted')
            return {"step_number": step + 1, "is_done": True}
        except asyncio.TimeoutError:
            agent.logger.error("LLM timeout")
            return {"step_number": step + 1, "consecutive_failures": failures + 1}
        except Exception as e:
            logger.error(f"Step error: {e}")
            agent.state.last_result = [ActionResult(error=str(e))]
            return {"step_number": step + 1, "consecutive_failures": failures + 1}
    
    def _should_continue(self, state: GraphState) -> Literal["continue", "done", "error"]:
        """Fast routing decision."""
        if state.get("is_done"):
            self.agent.logger.info('Task completed')
            return "done"
        if self.agent.state.stopped or self.agent.state.paused:
            return "done"
        if state.get("step_number", 0) >= state.get("max_steps", 100):
            return "done"
        if state.get("consecutive_failures", 0) >= self._max_failures:
            return "error"
        return "continue"
    
    async def run(self, max_steps: int = 100) -> AgentHistoryList:
        """Execute agent workflow via LangGraph."""
        state: GraphState = {
            "step_number": 0,
            "max_steps": max_steps,
            "is_done": False,
            "consecutive_failures": 0
        }
        await self.graph.ainvoke(state, config={"recursion_limit": max_steps + 10})
        
        if self.agent.history.is_done():
            await self.agent.log_completion()
            cb = self.agent.register_done_callback
            if cb:
                if inspect.iscoroutinefunction(cb):
                    await cb(self.agent.history)
                else:
                    cb(self.agent.history)
        
        return self.agent.history


def create_agent_graph(agent: 'Agent') -> AgentGraphBuilder:
    """Factory function to create agent graph builder."""
    return AgentGraphBuilder(agent)
