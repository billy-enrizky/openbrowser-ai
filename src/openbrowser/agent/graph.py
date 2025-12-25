"""LangGraph workflow for browser automation agent with Dynamic Re-Planning."""

import base64
import json
import logging
from typing import Annotated, Literal, TypedDict, List, Optional

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
    root_goal: str 
    plan: List[str] 
    current_step_index: int 

class BrowserAgent:
    """Browser automation agent using LangGraph workflow."""

    def __init__(self, headless: bool = True, model_name: str = "gpt-4o"):
        logger.info(f"Initializing BrowserAgent with model: {model_name}, headless: {headless}")
        
        self.browser_manager = BrowserManager(headless=headless)
        self.toolkit = BrowserToolKit(self.browser_manager)
        
        self.llm = ChatOpenAI(model=model_name, temperature=0, api_key=OPENAI_API_KEY)
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
            
            return {
                "screenshot": screenshot_result["data"],
                "dom_tree": dom_state.element_tree,
                "url": current_url,
            }
        finally:
            await client.stop()

    async def decompose_node(self, state: AgentState) -> dict:
        """Initial breakdown or Re-planning."""
        goal = state.get("root_goal") or state["messages"][0].content
        
        logger.info(f"Decomposing/Replanning goal: {goal}")
        planner_llm = self.llm.with_structured_output(TaskPlan)
        
        prompt = (
            f"You are a Browser Automation Strategist.\n"
            f"GOAL: {goal}\n\n"
            f"Create a step-by-step plan. Assume the browser is OPEN.\n"
            f"If there is a CAPTCHA or 'Sorry' page, include steps to handle it (e.g., 'Solve CAPTCHA')."
        )
        
        plan_result = await planner_llm.ainvoke(prompt)
        logger.info(f"New Plan: {plan_result.steps}")
        
        return {
            "root_goal": goal,
            "plan": plan_result.steps,
            "current_step_index": 0,
            "messages": [AIMessage(content=f"Plan updated: {plan_result.steps}")]
        }

    async def plan_node(self, state: AgentState) -> dict:
        """Dynamic Decision Node."""
        current_idx = state.get("current_step_index", 0)
        plan = state.get("plan", [])
        current_url = state.get("url", "")
        
        if current_idx >= len(plan):
            return {"messages": [AIMessage(content="DONE")]}

        current_task = plan[current_idx]
        logger.info(f"Processing Step {current_idx + 1}/{len(plan)}: {current_task}")

        # --- Dynamic System Prompt ---
        system_prompt = (
            "You are an intelligent browser automation agent.\n"
            f"GOAL: {state.get('root_goal')}\n"
            f"CURRENT STEP: {current_task}\n"
            f"URL: {current_url}\n"
            "DOM Tree elements are numbered [12].\n"
            "INSTRUCTIONS:\n"
            "1. If the current page matches the Current Step (e.g., Step='Navigate' but you are already there), SKIP it by responding 'NEXT STEP'.\n"
            "2. If you see a CAPTCHA/Sorry page: IGNORE the plan. Use tools to solve it (e.g., Click [Checkbox]).\n"
            "3. If you can perform the step, call the tool.\n"
            "4. If the plan is totally invalid for this page, respond 'REPLAN'.\n"
        )

        messages = [SystemMessage(content=system_prompt)]
        
        # Add filtered history
        for m in state["messages"][-5:]: # Keep last 5 messages for context
             if isinstance(m, HumanMessage) and "DOM Tree" in m.content: continue
             messages.append(m)

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
            logger.info(f"Skipping completed step: {current_task}")
            return {
                "current_step_index": current_idx + 1,
                "messages": [AIMessage(content="Moving to next step")]
            }
            
        return {"messages": [response]}

    async def execute_node(self, state: AgentState) -> dict:
        """Execute tools with Error Handling."""
        logger.info("Executing tool calls")
        messages = state["messages"]
        last_message = messages[-1]
        
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {"messages": []}
        
        tool_map = {tool.name: tool for tool in self.tools}
        tool_messages = []
        
        for tool_call in last_message.tool_calls:
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

    def _should_continue(self, state: AgentState) -> Literal["execute", "replan", "end"]:
        messages = state["messages"]
        if not messages: return "end"
        last_message = messages[-1]
        
        content = str(last_message.content)
        
        if "REPLANNING" in content:
            return "replan"
            
        if "Moving to next step" in content:
            # Loop back to Plan Node for the next step
            # We treat this as 'execute' flow which leads to perceive -> plan
            # But we need to ensure we don't just exit.
            # Actually, execute -> perceive -> plan. 
            # If we skip execution, we want to go straight to perceive.
            # The edge is plan -> execute. 
            # If we return 'execute', we go to execute_node (which does nothing if no tools) -> perceive -> plan.
            return "execute" 

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
                    "screenshot": "", "dom_tree": "", "url": "", 
                    "root_goal": goal, "plan": [], "current_step_index": 0
                }
                return await self.app.ainvoke(initial_state, config=config)
            finally:
                await client.stop()
        finally:
            await self.browser_manager.stop()
