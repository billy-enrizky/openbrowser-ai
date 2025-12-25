"""LangGraph workflow for browser automation agent with Planning and Memory."""

import base64
import json
import logging
from typing import Annotated, Literal, TypedDict, List

# CRITICAL FIX: Import add_messages
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from src.openbrowser.browser.dom import DomService
from src.openbrowser.browser.manager import BrowserManager
from src.openbrowser.tools.actions import BrowserToolKit

from dotenv import load_dotenv
import os
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")

logger = logging.getLogger(__name__)


# --- Data Models for Planning ---
class TaskPlan(BaseModel):
    """The broken-down tasks."""
    steps: List[str] = Field(description="List of sequential steps to achieve the goal")


class AgentState(TypedDict):
    """State for the browser automation agent."""

    # CRITICAL FIX: Use the actual function object, not a string
    messages: Annotated[list[BaseMessage], add_messages]
    screenshot: str  # Base64 encoded screenshot
    dom_tree: str  # Text representation of DOM for LLM
    # --- New Memory Fields ---
    root_goal: str  # Original user goal
    plan: List[str]  # List of decomposed subtasks
    current_step_index: int  # Current step being executed


class BrowserAgent:
    """Browser automation agent using LangGraph workflow.
    
    This agent orchestrates browser interactions through a perceive-plan-execute
    loop using LangGraph.
    """

    def __init__(self, headless: bool = True, model_name: str = "gpt-4o"):
        """Initialize BrowserAgent.
        
        Args:
            headless: Whether to run browser in headless mode
            model_name: OpenAI model name to use
        """
        logger.info(f"Initializing BrowserAgent with model: {model_name}, headless: {headless}")
        
        # Initialize browser manager and toolkit
        self.browser_manager = BrowserManager(headless=headless)
        self.toolkit = BrowserToolKit(self.browser_manager)
        
        # Initialize LLM
        self.llm = ChatOpenAI(model=model_name, temperature=0, api_key=OPENAI_API_KEY)
        
        # Bind tools to LLM
        self.tools = self.toolkit.get_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # --- Initialize Memory ---
        # This keeps the state in RAM during execution
        self.memory = MemorySaver()
        
        # Build and compile the graph
        self.app = self._build_graph(self.tools)
        
        logger.info("BrowserAgent initialized successfully")

    def _build_graph(self, tools: list) -> StateGraph:
        """Build the LangGraph workflow.
        
        Args:
            tools: List of LangChain tools to use
            
        Returns:
            Compiled StateGraph application
        """
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("decompose", self.decompose_node)
        workflow.add_node("perceive", self.perceive_node)
        workflow.add_node("plan", self.plan_node)
        workflow.add_node("execute", self.execute_node)
        
        # Start at Decomposition
        workflow.set_entry_point("decompose")
        
        # Flow: Decompose -> Perceive -> Plan -> Execute/Loop
        workflow.add_edge("decompose", "perceive")
        workflow.add_edge("perceive", "plan")
        workflow.add_conditional_edges(
            "plan",
            self._should_continue,
            {
                "continue": "execute",
                "end": END,
            },
        )
        workflow.add_edge("execute", "perceive")
        
        # Compile with checkpointer for memory retrieval
        return workflow.compile(checkpointer=self.memory)

    async def perceive_node(self, state: AgentState) -> dict:
        """Perceive the current browser state.
        
        Gets screenshot and DOM tree, updates toolkit state.
        
        Args:
            state: Current agent state
            
        Returns:
            State updates with screenshot and dom_tree
        """
        logger.info("Perceiving browser state")
        
        # Get CDP client and session
        client, session_id = await self.browser_manager.get_session()
        
        try:
            # Enable Page domain
            try:
                await client.send.Page.enable(session_id=session_id)
            except Exception:
                # Domain might already be enabled
                pass
            
            # Get screenshot as base64
            logger.info("Capturing screenshot")
            screenshot_result = await client.send.Page.captureScreenshot(
                params={"format": "png"}, session_id=session_id
            )
            screenshot_b64 = screenshot_result["data"]
            
            # Get DOM state
            logger.info("Extracting DOM state")
            dom_state = await DomService.get_clickable_elements(client, session_id)
            
            # CRITICAL: Update toolkit state to sync IDs
            self.toolkit.update_state(dom_state)
            
            logger.info(f"Perception complete: {len(dom_state.selector_map)} interactive elements found")
            
            return {
                "screenshot": screenshot_b64,
                "dom_tree": dom_state.element_tree,
            }
            
        finally:
            await client.stop()

    async def decompose_node(self, state: AgentState) -> dict:
        """Break the user's high-level goal into smaller steps.
        
        Uses structured output to generate a plan with sequential subtasks.
        
        Args:
            state: Current agent state with user goal in messages
            
        Returns:
            State updates with root_goal, plan, and current_step_index
        """
        logger.info("Decomposing task...")
        goal = state["messages"][0].content
        
        # Use structured output to force a JSON list
        planner_llm = self.llm.with_structured_output(TaskPlan)
        
        prompt = (
            f"Break this browser automation goal into clear, sequential steps.\n"
            f"Goal: {goal}\n\n"
            f"Keep steps atomic (e.g., 'Navigate to X', 'Click Y', 'Type Z')."
        )
        
        plan_result = await planner_llm.ainvoke(prompt)
        
        logger.info(f"Generated Plan: {plan_result.steps}")
        
        return {
            "root_goal": goal,
            "plan": plan_result.steps,
            "current_step_index": 0,
            # We add a system message to the history to set the stage
            "messages": [AIMessage(content=f"I have created a plan with {len(plan_result.steps)} steps.")]
        }

    async def plan_node(self, state: AgentState) -> dict:
        """Plan the next action based on current state.
        
        Decides next action based on the CURRENT subtask. Focuses the agent
        on one step at a time to prevent context bloat and maintain goal focus.
        
        Args:
            state: Current agent state
            
        Returns:
            State updates with LLM response messages or step advancement
        """
        current_idx = state.get("current_step_index", 0)
        plan = state.get("plan", [])
        
        # Safety check if plan is done
        if current_idx >= len(plan):
            return {"messages": [AIMessage(content="DONE")]}

        current_task = plan[current_idx]
        logger.info(f"Processing Step {current_idx + 1}/{len(plan)}: {current_task}")

        # --- Dynamic System Prompt ---
        # We inject ONLY the current task context to keep the LLM focused
        system_prompt = (
            "You are a precise browser automation agent.\n"
            f"OVERALL GOAL: {state.get('root_goal')}\n"
            f"YOUR CURRENT TASK: {current_task} (Step {current_idx+1} of {len(plan)})\n\n"
            "Interact with the page using the provided DOM tree [12].\n"
            "RULES:\n"
            "1. Focus ONLY on the Current Task.\n"
            "2. If the Current Task is finished, respond with 'NEXT STEP'.\n"
            "3. Only perform ONE action per turn.\n"
        )

        messages = [SystemMessage(content=system_prompt)]
        
        # Add recent history (Limit context window if needed, or rely on MemorySaver)
        # We filter the history to remove old DOM trees to keep it light
        for m in state["messages"]:
            if isinstance(m, (HumanMessage, AIMessage, ToolMessage)):
                # Hack: Don't re-send the huge DOM tree text in history, just the action
                if isinstance(m, HumanMessage) and "DOM Tree" in m.content:
                     # Only keep the latest perception (added below)
                     continue 
                messages.append(m)

        # Add current perception
        perception_content = f"Current page state:\n\nDOM Tree:\n{state['dom_tree']}"
        messages.append(HumanMessage(
            content=[
                {"type": "text", "text": perception_content},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{state['screenshot']}"}},
            ]
        ))
        
        response = await self.llm_with_tools.ainvoke(messages)
        
        # Check if the LLM thinks it's done with this subtask
        if "NEXT STEP" in str(response.content).upper():
            logger.info(f"Completed step: {current_task}")
            return {
                "current_step_index": current_idx + 1,
                "messages": [AIMessage(content=f"Completed: {current_task}. Moving to next step.")]
            }
            
        return {"messages": [response]}

    async def execute_node(self, state: AgentState) -> dict:
        """Execute tool calls from the last assistant message.
        
        Args:
            state: Current agent state
            
        Returns:
            State updates with ToolMessage responses
        """
        logger.info("Executing tool calls")
        
        messages = list(state["messages"])
        if not messages:
            logger.warning("No messages in state")
            return {"messages": []}
            
        last_message = messages[-1]
        
        # Get tool calls from last message (must be AIMessage)
        if not isinstance(last_message, AIMessage) or not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            logger.warning("Last message is not an AIMessage with tool_calls")
            return {"messages": []}
        
        # Create a tool map for quick lookup
        tool_map = {tool.name: tool for tool in self.tools}
        
        # Execute each tool call
        tool_messages = []
        for tool_call in last_message.tool_calls:
            # Handle different tool_call formats (dict or object)
            if isinstance(tool_call, dict):
                tool_name = tool_call.get("name") or tool_call.get("function", {}).get("name", "")
                tool_args = tool_call.get("args", {}) or tool_call.get("function", {}).get("arguments", {})
                tool_id = tool_call.get("id", "")
                # Parse arguments if it's a string (JSON)
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse tool args as JSON: {tool_args}")
                        tool_args = {}
            else:
                # Handle ToolCall object
                tool_name = getattr(tool_call, "name", None) or getattr(tool_call.function, "name", "") if hasattr(tool_call, "function") else ""
                tool_args = getattr(tool_call, "args", {})
                if not tool_args and hasattr(tool_call, "function"):
                    args_str = getattr(tool_call.function, "arguments", "{}")
                    if isinstance(args_str, str):
                        try:
                            tool_args = json.loads(args_str)
                        except json.JSONDecodeError:
                            tool_args = {}
                tool_id = getattr(tool_call, "id", "")
            
            if not tool_name:
                error_msg = "Tool call missing name"
                logger.error(error_msg)
                tool_messages.append(
                    ToolMessage(content=error_msg, tool_call_id=tool_id or "unknown")
                )
                continue
            
            logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
            
            if tool_name not in tool_map:
                error_msg = f"Tool {tool_name} not found"
                logger.error(error_msg)
                tool_messages.append(
                    ToolMessage(content=error_msg, tool_call_id=tool_id or "unknown")
                )
                continue
            
            try:
                # Get the tool and execute it (tools are async functions)
                tool = tool_map[tool_name]
                result = await tool.ainvoke(tool_args)
                
                logger.info(f"Tool {tool_name} executed successfully: {result[:100] if isinstance(result, str) else '...'}")
                tool_messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_id or "unknown")
                )
            except Exception as e:
                error_msg = f"Tool {tool_name} failed: {str(e)}"
                logger.error(error_msg, exc_info=True)
                tool_messages.append(
                    ToolMessage(content=error_msg, tool_call_id=tool_id or "unknown")
                )
        
        logger.info(f"Executed {len(tool_messages)} tool calls")
        return {"messages": tool_messages}

    def _should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        """Determine if we should continue or end.
        
        Checks if the last message is an AIMessage with tool calls, or if we
        need to move to the next step in the plan.
        
        Args:
            state: Current agent state
            
        Returns:
            "continue" if tool calls exist or moving to next step, "end" otherwise
        """
        messages = state["messages"]
        if not messages:
            return "end"
        
        last_message = messages[-1]
        
        # If we just moved to the next step (AIMessage with text only)
        if "Moving to next step" in str(last_message.content):
            # Check if we are actually done
            if state.get("current_step_index", 0) >= len(state.get("plan", [])):
                return "end"
            return "continue"  # Go back to perceive -> plan for the new task

        # If tool calls, execute them
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"
            
        return "end"

    async def run(self, goal: str) -> dict:
        """Run the agent with a goal.
        
        Starts browser, runs the graph, handles cleanup.
        
        Args:
            goal: User's goal/task description
            
        Returns:
            Final state from the graph execution
        """
        logger.info(f"Starting agent run with goal: {goal}")
        
        try:
            # Start browser
            await self.browser_manager.start()
            
            # Get initial CDP session for toolkit operations
            client, session_id = await self.browser_manager.get_session()
            
            try:
                # Configuration for the checkpointer
                config = {"configurable": {"thread_id": "1"}}
                
                # Create initial state
                initial_state: AgentState = {
                    "messages": [HumanMessage(content=goal)],
                    "screenshot": "",
                    "dom_tree": "",
                    "root_goal": "",
                    "plan": [],
                    "current_step_index": 0
                }
                
                # Pass config to ainvoke to use memory
                logger.info("Starting graph execution")
                final_state = await self.app.ainvoke(initial_state, config=config)
                
                logger.info("Graph execution completed")
                return final_state
                
            finally:
                await client.stop()
                
        finally:
            # Always stop browser
            await self.browser_manager.stop()
            logger.info("Browser stopped, agent run complete")

