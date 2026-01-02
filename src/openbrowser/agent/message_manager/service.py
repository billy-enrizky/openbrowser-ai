"""Message manager service for conversation history."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from langchain_core.messages import BaseMessage, SystemMessage

from src.openbrowser.agent.message_manager.views import (
    HistoryItem,
    MessageManagerState,
)
from src.openbrowser.agent.prompts import AgentMessagePrompt

if TYPE_CHECKING:
    from src.openbrowser.agent.views import ActionResult, AgentOutput, AgentStepInfo
    from src.openbrowser.browser.dom import DomState

logger = logging.getLogger(__name__)


class MessageManager:
    """Manages conversation history and state messages for the agent.
    
    Handles building and maintaining the message history that gets sent
    to the LLM, including system prompts, state messages, and context.
    
    Attributes:
        task: The current task description.
        state: The serializable message manager state.
        system_prompt: System message with agent instructions.
        max_history_items: Maximum history items to include in prompt.
        vision_detail_level: Detail level for screenshots ('auto', 'low', 'high').
        last_state_message_text: Text content of the last state message.
        
    Example:
        >>> manager = MessageManager(
        ...     task="Navigate to example.com",
        ...     system_message=system_msg,
        ...     max_history_items=20
        ... )
        >>> manager.create_state_messages(dom_state, url, screenshot)
        >>> messages = manager.get_messages()
    """

    def __init__(
        self,
        task: str,
        system_message: SystemMessage,
        state: MessageManagerState | None = None,
        max_history_items: int | None = None,
        vision_detail_level: Literal['auto', 'low', 'high'] = 'auto',
    ):
        self.task = task
        self.state = state if state is not None else MessageManagerState()
        self.system_prompt = system_message
        self.max_history_items = max_history_items
        self.vision_detail_level = vision_detail_level
        self.last_state_message_text: str | None = None

        # Initialize system message if state is empty
        if self.state.history.system_message is None:
            self.state.history.system_message = self.system_prompt

    @property
    def agent_history_description(self) -> str:
        """Build agent history description from list of items.
        
        Constructs a string representation of the agent's history,
        optionally omitting older items if max_history_items is set.
        
        Returns:
            String containing formatted history items.
        """
        if self.max_history_items is None:
            return '\n'.join(item.to_string() for item in self.state.agent_history_items)

        total_items = len(self.state.agent_history_items)
        if total_items <= self.max_history_items:
            return '\n'.join(item.to_string() for item in self.state.agent_history_items)

        # Omit older items
        omitted_count = total_items - self.max_history_items
        recent_items_count = self.max_history_items - 1

        items_to_include = [
            self.state.agent_history_items[0].to_string(),
            f'<sys>[... {omitted_count} previous steps omitted...]</sys>',
        ]
        items_to_include.extend([
            item.to_string() 
            for item in self.state.agent_history_items[-recent_items_count:]
        ])

        return '\n'.join(items_to_include)

    def add_new_task(self, new_task: str) -> None:
        """Add a follow-up task to the current conversation.
        
        Wraps the new task in XML tags and appends it to the existing
        task, also adding it to the agent history.
        
        Args:
            new_task: The follow-up task description.
        """
        new_task = f'<follow_up_request>{new_task.strip()}</follow_up_request>'
        if '<initial_request>' not in self.task:
            self.task = f'<initial_request>{self.task}</initial_request>'
        self.task += f'\n{new_task}'
        
        task_update_item = HistoryItem(system_message=new_task)
        self.state.agent_history_items.append(task_update_item)

    def _update_agent_history_description(
        self,
        model_output: 'AgentOutput | None' = None,
        result: 'list[ActionResult] | None' = None,
        step_info: 'AgentStepInfo | None' = None,
    ) -> None:
        """Update the agent history with results from the latest step.
        
        Processes action results and model output to create a new
        history item, handling content truncation and error formatting.
        
        Args:
            model_output: The agent's output from this step.
            result: List of action results from executed actions.
            step_info: Information about the current step.
        """
        if result is None:
            result = []
        
        step_number = step_info.step_number if step_info else None

        # Reset read state
        self.state.read_state_description = ''

        # Build action results string
        action_results = ''
        for action_result in result:
            if action_result.long_term_memory:
                action_results += f'{action_result.long_term_memory}\n'
            elif action_result.extracted_content and not action_result.include_extracted_content_only_once:
                action_results += f'{action_result.extracted_content}\n'
            
            if action_result.include_extracted_content_only_once and action_result.extracted_content:
                self.state.read_state_description += action_result.extracted_content + '\n'
            
            if action_result.error:
                error_text = action_result.error
                if len(error_text) > 200:
                    error_text = error_text[:100] + '...' + error_text[-100:]
                action_results += f'Error: {error_text}\n'

        # Truncate if too long
        max_content_size = 60000
        if len(self.state.read_state_description) > max_content_size:
            self.state.read_state_description = (
                self.state.read_state_description[:max_content_size] + 
                '\n... [Content truncated]'
            )

        if action_results:
            action_results = f'Result:\n{action_results}'
        action_results = action_results.strip() if action_results else None

        # Build history item
        if model_output is None:
            if step_number is not None:
                if step_number == 0 and action_results:
                    history_item = HistoryItem(
                        step_number=step_number, 
                        action_results=action_results
                    )
                    self.state.agent_history_items.append(history_item)
                elif step_number > 0:
                    history_item = HistoryItem(
                        step_number=step_number, 
                        error='Agent failed to output in the right format.'
                    )
                    self.state.agent_history_items.append(history_item)
        else:
            history_item = HistoryItem(
                step_number=step_number,
                evaluation_previous_goal=model_output.current_state.evaluation_previous_goal,
                memory=model_output.current_state.memory,
                next_goal=model_output.current_state.next_goal,
                action_results=action_results,
            )
            self.state.agent_history_items.append(history_item)

    def create_state_messages(
        self,
        dom_state: 'DomState',
        url: str,
        screenshot: str | None = None,
        model_output: 'AgentOutput | None' = None,
        result: 'list[ActionResult] | None' = None,
        step_info: 'AgentStepInfo | None' = None,
        use_vision: bool | Literal['auto'] = 'auto',
        action_descriptions: str | None = None,
    ) -> None:
        """Create state message with all current browser and agent context.
        
        Builds the complete state message to send to the LLM, including
        DOM state, URL, optional screenshot, and agent history.
        
        Args:
            dom_state: Current DOM state with interactive elements.
            url: Current page URL.
            screenshot: Optional base64-encoded screenshot.
            model_output: Agent's output from previous step.
            result: Action results from previous step.
            step_info: Current step information.
            use_vision: Whether to include screenshot (True, False, or 'auto').
            action_descriptions: Documentation for available actions.
        """
        # Clear context messages from previous steps
        self.state.history.context_messages.clear()

        # Update agent history
        self._update_agent_history_description(model_output, result, step_info)

        # Determine if we should include screenshot
        include_screenshot = False
        if use_vision is True:
            include_screenshot = True
        elif use_vision == 'auto':
            # In auto mode, include screenshot by default for better context
            include_screenshot = screenshot is not None

        # Create state message
        state_message = AgentMessagePrompt(
            task=self.task,
            dom_state=dom_state,
            url=url,
            screenshot=screenshot if include_screenshot else None,
            agent_history_description=self.agent_history_description,
            step_info=step_info,
            action_descriptions=action_descriptions,
            vision_detail_level=self.vision_detail_level,
        ).get_user_message(use_vision=include_screenshot)

        self.last_state_message_text = state_message.content if isinstance(state_message.content, str) else None
        self.state.history.state_message = state_message

    def get_messages(self) -> list[BaseMessage]:
        """Get the current message list for sending to the LLM.
        
        Returns:
            List of BaseMessage instances in proper order.
        """
        return self.state.history.get_messages()

    def add_context_message(self, message: BaseMessage) -> None:
        """Add a contextual message for this step.
        
        Context messages are cleared at the start of each new step
        and are used for step-specific additional context.
        
        Args:
            message: The context message to add.
        """
        self.state.history.context_messages.append(message)

