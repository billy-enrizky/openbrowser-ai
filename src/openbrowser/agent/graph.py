"""LangGraph workflow for browser automation agent with Dynamic Re-Planning."""

import asyncio
import base64
import json
import logging
import re
from typing import Annotated, Literal, TypedDict, List, Optional

from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from src.openbrowser.browser.dom import DomService
from src.openbrowser.browser.manager import BrowserManager
from src.openbrowser.tools.actions import BrowserToolKit
from src.openbrowser.llm.google import ChatGoogle
from src.openbrowser.llm.openai import ChatOpenAI

from dotenv import load_dotenv
import os
load_dotenv(override=True)

logger = logging.getLogger(__name__)

# --- Data Models ---
class TaskPlan(BaseModel):
    """The broken-down tasks."""
    steps: List[str] = Field(description="List of sequential steps to achieve the goal")

class RePlan(BaseModel):
    """Dynamic update to the plan."""
    reasoning: str = Field(description="Why the plan needs to change")
    new_steps: List[str] = Field(description="The new remaining steps to execute")
    is_done: bool = Field(description="Whether the entire goal is achieved")

class AgentState(TypedDict):
    """State for the browser automation agent."""
    messages: Annotated[list[BaseMessage], add_messages]
    screenshot: str 
    dom_tree: str 
    url: str 
    previous_url: str  # Track URL changes to detect navigation completion
    root_goal: str 
    plan: List[str] 
    current_step_index: int
    step_attempt_count: int  # Track retries on same step to detect loops
    google_failure_count: int  # Track consecutive failures on Google to trigger DuckDuckGo fallback 

class BrowserAgent:
    """Browser automation agent using LangGraph workflow."""

    def __init__(
        self,
        headless: bool = True,
        model_name: str = "gpt-4o",
        llm_provider: str = "openai",
        api_key: str | None = None,
    ):
        """
        Initialize BrowserAgent.

        Args:
            headless: Whether to run browser in headless mode
            model_name: Name of the model to use (e.g., "gpt-4o", "gemini-flash-latest")
            llm_provider: LLM provider to use ("openai" or "google")
            api_key: API key for the LLM provider (defaults to environment variables)
        """
        logger.info(f"Initializing BrowserAgent with provider: {llm_provider}, model: {model_name}, headless: {headless}")
        
        self.browser_manager = BrowserManager(headless=headless)
        self.toolkit = BrowserToolKit(self.browser_manager)
        
        if llm_provider == "google":
            if api_key:
                self.llm = ChatGoogle(model=model_name, temperature=0, api_key=api_key)
            else:
                self.llm = ChatGoogle(model=model_name, temperature=0)
        else:
            self.llm = ChatOpenAI(model=model_name, temperature=0, api_key=api_key)
        
        self.tools = self.toolkit.get_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.memory = MemorySaver()
        self.app = self._build_graph(self.tools)

    def _build_graph(self, tools: list) -> StateGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node("decompose", self.decompose_node)
        workflow.add_node("perceive", self.perceive_node)
        workflow.add_node("plan", self.plan_node)
        workflow.add_node("execute", self.execute_node)
        
        workflow.set_entry_point("decompose")
        
        workflow.add_edge("decompose", "perceive")
        workflow.add_edge("perceive", "plan")
        workflow.add_conditional_edges(
            "plan",
            self._should_continue,
            {
                "execute": "execute",
                "next_step": "plan",  # Step completed, loop back to plan for next step
                "continue": "perceive",  # Step completed, go directly to perceive for next step
                "replan": "decompose", # If plan fails, re-decompose
                "end": END,
            },
        )
        workflow.add_edge("execute", "perceive")
        return workflow.compile(checkpointer=self.memory)

    async def perceive_node(self, state: AgentState) -> dict:
        logger.info("Perceiving browser state")
        client, session_id = await self.browser_manager.get_session()
        try:
            try: await client.send.Page.enable(session_id=session_id)
            except Exception: pass
            
            # Wait for page to stabilize after potential navigation
            # Check if page is still loading and wait if needed
            try:
                # Get current frame tree to check loading state
                frame_tree = await client.send.Page.getFrameTree(session_id=session_id)
                # Small delay to ensure DOM is stable after navigation
                await asyncio.sleep(0.8)
            except Exception:
                # If check fails, just wait a bit
                await asyncio.sleep(0.8)
            
            screenshot_result = await client.send.Page.captureScreenshot(params={"format": "png"}, session_id=session_id)
            dom_state = await DomService.get_clickable_elements(client, session_id)
            self.toolkit.update_state(dom_state)
            
            current_url = ""
            try:
                nav_history = await client.send.Page.getNavigationHistory(session_id=session_id)
                idx = nav_history.get("currentIndex", 0)
                entries = nav_history.get("entries", [])
                if entries and idx < len(entries):
                    current_url = entries[idx].get("url", "")
            except Exception: pass
            
            previous_url = state.get("url", "")
            return {
                "screenshot": screenshot_result["data"],
                "dom_tree": dom_state.element_tree,
                "url": current_url,
                "previous_url": previous_url,
            }
        finally:
            await client.stop()

    def _is_google_traffic_verification(self, url: str, dom_tree: str) -> bool:
        """Check if we're on a Google traffic verification page."""
        if not url:
            return False
        
        # Check URL patterns for Google traffic verification
        google_verification_patterns = [
            "google.com/sorry",
            "google.com/check",
            "consent.google.com",
        ]
        
        url_lower = url.lower()
        if any(pattern in url_lower for pattern in google_verification_patterns):
            return True
        
        # Check DOM for traffic verification indicators
        dom_lower = dom_tree.lower()
        verification_keywords = [
            "traffic verification",
            "verify you're not a robot",
            "unusual traffic",
            "sorry, we have detected unusual traffic",
        ]
        
        if any(keyword in dom_lower for keyword in verification_keywords):
            return True
        
        return False

    async def decompose_node(self, state: AgentState) -> dict:
        """Initial breakdown or Re-planning."""
        goal = state.get("root_goal") or state["messages"][0].content
        
        logger.info(f"Decomposing/Replanning goal: {goal}")
        
        try:
            planner_llm = self.llm.with_structured_output(TaskPlan)
            
            prompt = (
                f"You are a Browser Automation Strategist.\n"
                f"GOAL: {goal}\n\n"
                f"Create a step-by-step plan. Assume the browser is OPEN.\n"
                f"Note: The agent can handle unexpected pages (CAPTCHA, login, errors, etc.) dynamically during execution, so you don't need to pre-plan for every possible obstacle."
            )
            
            plan_result = await planner_llm.ainvoke(prompt)
            
            # Validate plan_result
            if plan_result is None:
                raise ValueError("Structured output returned None")
            
            if not hasattr(plan_result, 'steps') or not plan_result.steps:
                raise ValueError(f"Invalid plan result: {plan_result}")
            
            logger.info(f"New Plan: {plan_result.steps}")
            
            return {
                "root_goal": goal,
                "plan": plan_result.steps,
                "current_step_index": 0,
                "step_attempt_count": 0,
                "google_failure_count": state.get("google_failure_count", 0),
                "previous_url": state.get("url", ""),
                "messages": [AIMessage(content=f"Plan updated: {plan_result.steps}")]
            }
        except Exception as e:
            logger.error(f"Failed to get structured output: {e}. Falling back to text-based planning.")
            
            # Fallback: Use regular LLM call and parse response
            prompt = (
                f"You are a Browser Automation Strategist.\n"
                f"GOAL: {goal}\n\n"
                f"Create a step-by-step plan. Assume the browser is OPEN.\n"
                f"Respond with a JSON object containing a 'steps' array of strings.\n"
                f"Example: {{\"steps\": [\"Step 1\", \"Step 2\", \"Step 3\"]}}\n"
                f"Note: The agent can handle unexpected pages (CAPTCHA, login, errors, etc.) dynamically during execution, so you don't need to pre-plan for every possible obstacle."
            )
            
            try:
                response = await self.llm.ainvoke([HumanMessage(content=prompt)])
                content = response.content if hasattr(response, 'content') else str(response)
                
                # Try to extract JSON from response
                json_match = re.search(r'\{[^{}]*"steps"[^{}]*\[[^\]]*\][^{}]*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed = json.loads(json_str)
                    steps = parsed.get("steps", [])
                else:
                    # Fallback: try to parse as simple list
                    lines = [line.strip() for line in content.split('\n') if line.strip() and not line.strip().startswith('#')]
                    steps = [line for line in lines if line and not line.startswith('{') and not line.startswith('[')]
                    if not steps:
                        # Last resort: create a simple plan
                        steps = [
                            f"Navigate to the website mentioned in the goal",
                            f"Complete the task: {goal}",
                        ]
                
                logger.info(f"Fallback Plan: {steps}")
                
                return {
                    "root_goal": goal,
                    "plan": steps,
                    "current_step_index": 0,
                    "step_attempt_count": 0,
                    "google_failure_count": state.get("google_failure_count", 0),
                    "previous_url": state.get("url", ""),
                    "messages": [AIMessage(content=f"Plan updated (fallback): {steps}")]
                }
            except Exception as fallback_error:
                logger.error(f"Fallback planning also failed: {fallback_error}")
                # Last resort: create a minimal plan
                steps = [
                    f"Navigate to complete the goal: {goal}",
                    "Complete the required actions",
                ]
                return {
                    "root_goal": goal,
                    "plan": steps,
                    "current_step_index": 0,
                    "step_attempt_count": 0,
                    "google_failure_count": state.get("google_failure_count", 0),
                    "previous_url": state.get("url", ""),
                    "messages": [AIMessage(content=f"Plan created (minimal fallback): {steps}")]
                }

    async def plan_node(self, state: AgentState) -> dict:
        """Dynamic Decision Node."""
        current_idx = state.get("current_step_index", 0)
        plan = state.get("plan", [])
        current_url = state.get("url", "")
        previous_url = state.get("previous_url", "")
        attempts = state.get("step_attempt_count", 0)
        google_failures = state.get("google_failure_count", 0)
        dom_tree = state.get("dom_tree", "")
        
        # Check if we're on DuckDuckGo - if so, skip Google verification checks
        is_on_duckduckgo = "duckduckgo.com" in current_url.lower() if current_url else False
        
        # Check for Google traffic verification page (only if not on DuckDuckGo)
        is_google_verification = False
        is_on_google = False
        if not is_on_duckduckgo:
            is_google_verification = self._is_google_traffic_verification(current_url, dom_tree)
            is_on_google = "google.com" in current_url.lower() if current_url else False
        
        # Reset Google failure count if we're no longer on Google (successful navigation away)
        new_failure_count = google_failures
        if not is_on_google and google_failures > 0:
            new_failure_count = 0
            logger.info("No longer on Google, resetting failure count")
        
        # Increment Google failure count if we hit verification page (only check if on Google)
        if is_google_verification and is_on_google and not is_on_duckduckgo:
            new_failure_count = google_failures + 1
            logger.warning(f"Detected Google traffic verification page (failure {new_failure_count}/3)")
        
        # If we hit Google traffic verification 3 times, switch to DuckDuckGo
        if is_google_verification and is_on_google and new_failure_count >= 3 and not is_on_duckduckgo:
            logger.warning(f"Google failed {new_failure_count} times. Switching to DuckDuckGo.")
            # Navigate to DuckDuckGo
            client, session_id = await self.browser_manager.get_session()
            try:
                await self.toolkit.navigate("https://duckduckgo.com", client=client, session_id=session_id)
                logger.info("Navigated to DuckDuckGo as fallback")
                # Wait for navigation to complete
                await asyncio.sleep(1.5)
            finally:
                await client.stop()
            
            # Update plan to use DuckDuckGo instead of Google
            updated_plan = []
            for step in plan:
                # Replace Google references with DuckDuckGo
                updated_step = step.replace("Google", "DuckDuckGo").replace("google.com", "duckduckgo.com")
                updated_plan.append(updated_step)
            
            # Route to perceive to get fresh state after navigation
            # Don't return early - let the normal flow continue but skip Google check
            # We'll update the plan and let perceive get the new DuckDuckGo state
            return {
                "plan": updated_plan,
                "google_failure_count": 0,  # Reset counter after switching
                "step_attempt_count": 0,  # Reset step attempts since we're switching strategy
                "messages": [AIMessage(content="Switched to DuckDuckGo after 3 Google failures. Plan updated.")]
            }
        
        # 1. Handle End of Plan
        if current_idx >= len(plan):
            logger.info("Plan completed, all steps done")
            return {"messages": [AIMessage(content="DONE")]}

        current_task = plan[current_idx]
        logger.info(f"Processing Step {current_idx + 1}/{len(plan)}: {current_task} (Attempt {attempts + 1})")
        logger.debug(f"Current URL: {current_url}, Previous URL: {previous_url}")
        
        # Check if URL changed (navigation happened) - this might indicate step completion
        url_changed = current_url != previous_url and previous_url != ""

        # 2. Refined Prompt to prevent "Skipping Ahead" and force step completion validation
        url_context = ""
        if url_changed:
            url_context = f"\nNOTE: The URL changed from '{previous_url}' to '{current_url}'. This may indicate navigation completed."
        
        # Loop detection warning
        loop_warning = ""
        if attempts >= 3:
            loop_warning = f"\nWARNING: You have attempted this step {attempts + 1} times. If you are stuck, reply 'REPLAN' to regenerate the plan."
        
        # Google failure warning
        google_warning = ""
        if is_google_verification and is_on_google:
            google_warning = f"\nWARNING: Google traffic verification detected (failure {new_failure_count}/3). If this persists, the agent will automatically switch to DuckDuckGo."
        
        # DuckDuckGo fallback instruction
        fallback_note = ""
        if new_failure_count >= 3:
            fallback_note = "\nNOTE: After 3 consecutive failures on Google, the agent will automatically switch to DuckDuckGo for searches."
        
        system_prompt = (
            f"ROOT GOAL: {state.get('root_goal')}\n"
            f"CURRENT PLAN STEP {current_idx + 1}/{len(plan)}: {current_task}\n"
            f"CURRENT URL: {current_url}{url_context}{loop_warning}{google_warning}{fallback_note}\n\n"
            "INSTRUCTIONS:\n"
            "1. LOOK at the screenshot and URL.\n"
            "2. IS THIS STEP ALREADY COMPLETED? (e.g., if step is 'Go to Google' and you are on Google, or if step is 'Type text' and text is already typed, or if step is 'Click first result' and you just clicked it).\n"
            "   - If YES: Reply exactly 'NEXT STEP'. Do not use tools.\n"
            "3. IF NOT completed: Generate the correct Tool Call to perform it.\n"
            "4. IF stuck (same step > 3 times) or CAPTCHA appears: Reply 'REPLAN'.\n"
            "5. AFTER EXECUTING A TOOL: Check if the tool execution completed the current step. If yes, reply 'NEXT STEP' in your next response.\n"
            "6. UNEXPECTED PAGES: If you encounter an unexpected page state (CAPTCHA, login page, error page, cookie consent, pop-up, etc.), handle it directly using tools. Do not wait for the plan to address it.\n"
            "7. GOOGLE FAILURES: If you encounter Google traffic verification pages 3 times in a row, the agent will automatically switch to DuckDuckGo. You can also proactively use DuckDuckGo if Google is blocking access.\n"
            "8. DO NOT combine steps. Finish the current step before moving to the next.\n"
            "DOM Tree elements are numbered [1], [2], [3], etc. starting from 1. Use these numbers to interact with elements via the tools."
        )

        messages = [SystemMessage(content=system_prompt)]
        
        # Add filtered history - preserve AIMessage-ToolMessage pairs
        # OpenAI requires ToolMessages to immediately follow their AIMessage with tool_calls
        # CRITICAL: ALL tool_call_ids in an AIMessage must have corresponding ToolMessages
        history = state["messages"]
        filtered_history = []
        
        # Start from the end and work backwards, keeping pairs together
        i = len(history) - 1
        while i >= 0 and len(filtered_history) < 10:  # Keep up to 10 messages
            msg = history[i]
            
            # Skip DOM tree HumanMessages
            if isinstance(msg, HumanMessage) and "DOM Tree" in str(msg.content):
                i -= 1
                continue
            
            # If it's a ToolMessage, collect all consecutive ToolMessages and check if they match an AIMessage
            if isinstance(msg, ToolMessage):
                # Collect all consecutive ToolMessages starting from current position
                tool_messages = []
                j = i
                while j >= 0 and isinstance(history[j], ToolMessage):
                    tool_messages.insert(0, history[j])  # Insert at beginning to maintain order
                    j -= 1
                
                # Check if the message before the ToolMessages is an AIMessage with tool_calls
                if j >= 0 and isinstance(history[j], AIMessage):
                    ai_msg = history[j]
                    if ai_msg.tool_calls:
                        # Get all tool_call_ids from the AIMessage
                        ai_tool_call_ids = {tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None) for tc in ai_msg.tool_calls}
                        ai_tool_call_ids = {tid for tid in ai_tool_call_ids if tid is not None}
                        
                        # Get all tool_call_ids from the ToolMessages
                        tool_msg_ids = {tm.tool_call_id for tm in tool_messages if hasattr(tm, "tool_call_id")}
                        
                        # Only include if ALL tool_call_ids have corresponding ToolMessages
                        if ai_tool_call_ids == tool_msg_ids and len(ai_tool_call_ids) > 0:
                            # All tool calls have responses - include both AIMessage and all ToolMessages
                            # Insert AIMessage at position 0, then insert ToolMessages right after it
                            filtered_history.insert(0, ai_msg)  # AIMessage first
                            for idx, tm in enumerate(tool_messages):
                                filtered_history.insert(1 + idx, tm)  # Insert ToolMessages right after AIMessage
                            i = j - 1  # Skip the AIMessage and all ToolMessages
                        else:
                            # Not all tool calls have responses - skip the entire AIMessage and ToolMessages
                            # This prevents OpenAI API error about missing tool responses
                            i = j - 1
                    else:
                        # AIMessage doesn't have tool_calls - orphaned ToolMessages, skip them
                        i = j
                else:
                    # No preceding AIMessage - orphaned ToolMessages, skip them
                    i = j
            elif isinstance(msg, AIMessage) and msg.tool_calls:
                # This is an AIMessage with tool_calls, but we haven't seen ToolMessages yet
                # Check if there are ToolMessages after this (forward in history)
                # Since we're going backwards, if we hit an AIMessage with tool_calls,
                # we should have already processed its ToolMessages above
                # If we're here, it means the ToolMessages are missing - skip this AIMessage
                i -= 1
            else:
                # Regular message (AIMessage without tool_calls, HumanMessage, SystemMessage, etc.)
                filtered_history.insert(0, msg)
                i -= 1
        
        messages.extend(filtered_history)

        # Add perception
        perception = f"URL: {current_url}\nDOM:\n{state['dom_tree']}"
        messages.append(HumanMessage(
            content=[
                {"type": "text", "text": perception},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{state['screenshot']}"}},
            ]
        ))
        
        response = await self.llm_with_tools.ainvoke(messages)
        content = str(response.content).upper()

        if "REPLAN" in content:
            logger.warning("Agent requested replanning.")
            return {"messages": [AIMessage(content="REPLANNING")]}
            
        if "NEXT STEP" in content:
            new_idx = current_idx + 1
            logger.info(f"Step {current_idx + 1} marked complete by Agent. Advancing to step {new_idx + 1}")
            # Reset Google failure count on successful step completion
            return {
                "current_step_index": new_idx,
                "step_attempt_count": 0,  # Reset attempt count for new step
                "google_failure_count": 0,  # Reset Google failures on successful step
                "messages": [AIMessage(content=f"Completed step {current_idx + 1}. Moving to {new_idx + 1}.")]
            }
        
        # Update Google failure count if we're on Google verification page
        update_dict = {
            "messages": [response],
            "step_attempt_count": attempts + 1
        }
        if is_google_verification and is_on_google:
            update_dict["google_failure_count"] = new_failure_count
            
        # Handle tool calls - increment attempt count
        return update_dict

    async def execute_node(self, state: AgentState) -> dict:
        """Execute tools with Error Handling."""
        logger.info("Executing tool calls")
        messages = state["messages"]
        last_message = messages[-1]
        
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {"messages": []}
        
        # Use get_tools_map() if available, otherwise fall back to dict comprehension
        if hasattr(self.toolkit, 'get_tools_map'):
            tool_map = self.toolkit.get_tools_map()
        else:
            tool_map = {tool.name: tool for tool in self.tools}
        
        tool_messages = []
        
        # Tool calls are in LangChain format: {"id": "...", "name": "...", "args": {...}}
        # This matches the format produced by both ChatOpenAI and ChatGoogle
        for tool_call in last_message.tool_calls:
            # Handle both flat format (new) and nested format (old) for backward compatibility
            if "function" in tool_call and isinstance(tool_call["function"], dict):
                # Old nested format: {"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}
                tool_name = tool_call["function"]["name"]
                tool_args_str = tool_call["function"].get("arguments", "{}")
                try:
                    tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
                except (json.JSONDecodeError, TypeError):
                    tool_args = {}
                tool_id = tool_call["id"]
            else:
                # New flat format: {"id": "...", "name": "...", "args": {...}}
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
            
            logger.info(f"Executing: {tool_name} {tool_args}")
            
            try:
                tool = tool_map[tool_name]
                result = await tool.ainvoke(tool_args)
                tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))
            except Exception as e:
                # CRITICAL: Feed error back to LLM so it can retry
                error_msg = f"Error executing {tool_name}: {str(e)}"
                logger.error(error_msg)
                tool_messages.append(ToolMessage(content=error_msg, tool_call_id=tool_id))
        
        return {"messages": tool_messages}

    def _should_continue(self, state: AgentState) -> Literal["execute", "next_step", "continue", "replan", "end"]:
        messages = state["messages"]
        if not messages: return "end"
        last_message = messages[-1]
        
        content = str(last_message.content)
        
        if "REPLANNING" in content:
            return "replan"
        
        # If we switched to DuckDuckGo, route to perceive to get fresh state
        if "Switched to DuckDuckGo" in content:
            return "continue"
            
        # If the last message was a text "Completed step...", we need to loop back to PLAN
        if isinstance(last_message, AIMessage) and not last_message.tool_calls and "Completed step" in content:
            return "next_step"
            
        if "Step" in content and "completed" in content:
            # Step was marked complete, go directly to perceive for next step
            # Skip execute since there are no tool calls
            return "continue" 

        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "execute"
            
        if "DONE" in content:
            return "end"

        return "end"

    async def run(self, goal: str) -> dict:
        logger.info(f"Starting agent run: {goal}")
        try:
            await self.browser_manager.start()
            client, session_id = await self.browser_manager.get_session()
            try:
                config = {"configurable": {"thread_id": "1"}}
                initial_state = {
                    "messages": [HumanMessage(content=goal)],
                    "screenshot": "", "dom_tree": "", "url": "", "previous_url": "",
                    "root_goal": goal, "plan": [], "current_step_index": 0, "step_attempt_count": 0,
                    "google_failure_count": 0
                }
                return await self.app.ainvoke(initial_state, config={**config, "recursion_limit": 100})
            finally:
                await client.stop()
        finally:
            await self.browser_manager.stop()
