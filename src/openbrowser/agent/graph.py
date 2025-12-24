"""LangGraph workflow for browser automation agent."""

import base64
import json
import logging
from typing import Annotated, Literal, TypedDict

# CRITICAL FIX: Import add_messages
from langgraph.graph.message import add_messages

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

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


class AgentState(TypedDict):
    """State for the browser automation agent."""

    # CRITICAL FIX: Use the actual function object, not a string
    messages: Annotated[list[BaseMessage], add_messages]
    screenshot: str  # Base64 encoded screenshot
    dom_tree: str  # Text representation of DOM for LLM


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
        workflow.add_node("perceive", self.perceive_node)
        workflow.add_node("plan", self.plan_node)
        workflow.add_node("execute", self.execute_node)
        
        # Set entry point
        workflow.set_entry_point("perceive")
        
        # Add edges
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
        
        # Compile the graph
        return workflow.compile()

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

    async def plan_node(self, state: AgentState) -> dict:
        """Plan the next action based on current state.
        
        Constructs messages with screenshot and DOM tree, calls LLM.
        
        Args:
            state: Current agent state
            
        Returns:
            State updates with LLM response messages
        """
        logger.info("Planning next action")
        
        # Get all messages from state
        messages = list(state["messages"])
        
        # Check if system message already exists
        has_system_message = any(isinstance(msg, SystemMessage) for msg in messages)
        
        # Build messages for LLM call
        llm_messages = []
        
        # Add system message if not present
        if not has_system_message:
            system_message = SystemMessage(
                content=(
                    "You are a precise browser automation agent. "
                    "Interact with the page using the provided DOM tree where elements are numbered like [12]. "
                    "Use tools to navigate, click, and type. "
                    "\n\nRULES:\n"
                    "1. CRITICAL: Only perform ONE action per turn. Never call multiple tools at once.\n"
                    "2. If you click a link that changes the page, stop and wait for the next turn.\n"
                    "3. If the goal is achieved, simply respond with 'DONE'.\n"
                    "4. If you encounter an error, try a different approach."
                )
            )
            llm_messages.append(system_message)
        else:
            # Include existing system message
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    llm_messages.append(msg)
                    break
        
        # Build conversation history, ensuring ToolMessages follow their AIMessage
        # OpenAI API requires: AIMessage (with tool_calls) -> ToolMessage(s)
        # Process messages in order, pairing AIMessages with their ToolMessages
        i = 0
        while i < len(messages):
            msg = messages[i]
            
            # Skip system messages (already added)
            if isinstance(msg, SystemMessage):
                i += 1
                continue
            
            # If it's an AIMessage, check if it has tool_calls
            if isinstance(msg, AIMessage):
                llm_messages.append(msg)
                i += 1
                # If it has tool_calls, include all following ToolMessages
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    while i < len(messages) and isinstance(messages[i], ToolMessage):
                        llm_messages.append(messages[i])
                        i += 1
            # Include HumanMessages
            elif isinstance(msg, HumanMessage):
                llm_messages.append(msg)
                i += 1
            # Skip orphaned ToolMessages (shouldn't happen in normal flow)
            elif isinstance(msg, ToolMessage):
                logger.warning(f"Skipping orphaned ToolMessage at index {i} (no preceding AIMessage with tool_calls)")
                i += 1
            else:
                # Include other message types
                llm_messages.append(msg)
                i += 1
        
        # Add current perception (screenshot + DOM tree) as a new human message
        perception_content = f"Current page state:\n\nDOM Tree:\n{state['dom_tree']}"
        perception_message = HumanMessage(
            content=[
                {"type": "text", "text": perception_content},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{state['screenshot']}"},
                },
            ]
        )
        llm_messages.append(perception_message)
        
        # Call LLM with tools
        response = await self.llm_with_tools.ainvoke(llm_messages)
        
        logger.info(f"LLM response: {response.content}")
        if hasattr(response, "tool_calls") and response.tool_calls:
            logger.info(f"Tool calls: {len(response.tool_calls)}")
        
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
        
        Checks if the last message is an AIMessage with tool calls.
        
        Args:
            state: Current agent state
            
        Returns:
            "continue" if tool calls exist, "end" otherwise
        """
        messages = state["messages"]
        if not messages:
            logger.info("No messages, ending workflow")
            return "end"
            
        last_message = messages[-1]
        
        # Check if last message is an AIMessage with tool calls
        if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            logger.info("Tool calls detected, continuing to execute")
            return "continue"
        
        logger.info("No tool calls, ending workflow")
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
                # Create initial state
                initial_state: AgentState = {
                    "messages": [HumanMessage(content=goal)],
                    "screenshot": "",
                    "dom_tree": "",
                }
                
                # Run the graph
                logger.info("Starting graph execution")
                final_state = await self.app.ainvoke(initial_state)
                
                logger.info("Graph execution completed")
                return final_state
                
            finally:
                await client.stop()
                
        finally:
            # Always stop browser
            await self.browser_manager.stop()
            logger.info("Browser stopped, agent run complete")

