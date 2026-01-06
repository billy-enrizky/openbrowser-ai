"""Prompt classes for agent system and user messages."""

from __future__ import annotations

import importlib.resources
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from langchain_core.messages import HumanMessage, SystemMessage

if TYPE_CHECKING:
    from openbrowser.agent.views import AgentStepInfo
    from openbrowser.browser.dom import DomState


class SystemPrompt:
    """System prompt loaded from markdown template.
    
    Manages the system prompt that defines the agent's behavior and
    instructions. Loads from markdown templates with support for
    customization via override or extension.
    
    Attributes:
        max_actions_per_step: Maximum actions allowed per step.
        use_thinking: Whether thinking mode is enabled.
        flash_mode: Whether flash mode (minimal output) is enabled.
        system_message: The compiled SystemMessage instance.
        
    Example:
        >>> prompt = SystemPrompt(
        ...     max_actions_per_step=4,
        ...     override_system_message="Custom instructions..."
        ... )
        >>> message = prompt.get_system_message()
    """

    def __init__(
        self,
        max_actions_per_step: int = 4,
        override_system_message: str | None = None,
        extend_system_message: str | None = None,
        use_thinking: bool = True,
        flash_mode: bool = False,
    ):
        self.max_actions_per_step = max_actions_per_step
        self.use_thinking = use_thinking
        self.flash_mode = flash_mode
        
        prompt = ''
        if override_system_message is not None:
            prompt = override_system_message
        else:
            self._load_prompt_template()
            prompt = self.prompt_template.format(max_actions=self.max_actions_per_step)

        if extend_system_message:
            prompt += f'\n{extend_system_message}'

        self.system_message = SystemMessage(content=prompt)

    def _load_prompt_template(self) -> None:
        """Load the prompt template from the markdown file.
        
        Loads the appropriate template based on use_thinking and flash_mode
        settings. Falls back to a basic prompt if the file is not found.
        """
        try:
            if self.flash_mode:
                template_filename = 'system_prompt_flash.md'
            elif self.use_thinking:
                template_filename = 'system_prompt.md'
            else:
                template_filename = 'system_prompt_no_thinking.md'

            # Load from the same directory as this module
            with importlib.resources.files('openbrowser.agent').joinpath(template_filename).open('r', encoding='utf-8') as f:
                self.prompt_template = f.read()
        except Exception as e:
            # Fallback to a basic prompt if file not found
            self.prompt_template = self._get_fallback_prompt()

    def _get_fallback_prompt(self) -> str:
        """Return a fallback prompt if template file is not found.
        
        Provides a basic but functional system prompt when the template
        files are not available.
        
        Returns:
            Basic system prompt string with {max_actions} placeholder.
        """
        return """You are an AI agent designed to automate browser tasks.

Your goal is to complete the user's request by interacting with web pages.

At each step you receive:
1. Agent history with previous actions and results
2. Current browser state with URL and interactive elements
3. A screenshot of the current page

You must respond with valid JSON containing:
- thinking: Your reasoning about what to do
- evaluation_previous_goal: Assessment of last action (Success/Failure)
- memory: Important context to remember
- next_goal: Your immediate next objective
- action: List of actions to execute (max {max_actions} per step)

Interactive elements are shown as [index]<type>text</type>.
Only use numeric indexes that are explicitly provided.

When done, use the done action with your result."""

    def get_system_message(self) -> SystemMessage:
        """Get the system prompt message.
        
        Returns:
            SystemMessage instance with the compiled prompt.
        """
        return self.system_message


class AgentMessagePrompt:
    """Builds user message with browser state, history, and task.
    
    Constructs the user message sent to the LLM at each step, combining
    the current browser state, agent history, task description, and
    optionally a screenshot.
    
    Attributes:
        task: The current task description.
        dom_state: Current DOM state with interactive elements.
        url: Current page URL.
        screenshot: Optional base64 screenshot.
        agent_history_description: Summary of previous steps.
        step_info: Current step number and max steps.
        action_descriptions: Available actions documentation.
        vision_detail_level: Screenshot detail level.
        
    Example:
        >>> prompt = AgentMessagePrompt(
        ...     task="Find contact info",
        ...     dom_state=dom_state,
        ...     url="https://example.com",
        ...     screenshot=screenshot_b64
        ... )
        >>> message = prompt.get_user_message(use_vision=True)
    """

    def __init__(
        self,
        task: str,
        dom_state: 'DomState',
        url: str,
        screenshot: str | None = None,
        agent_history_description: str | None = None,
        step_info: 'AgentStepInfo | None' = None,
        action_descriptions: str | None = None,
        vision_detail_level: Literal['auto', 'low', 'high'] = 'auto',
    ):
        self.task = task
        self.dom_state = dom_state
        self.url = url
        self.screenshot = screenshot
        self.agent_history_description = agent_history_description
        self.step_info = step_info
        self.action_descriptions = action_descriptions
        self.vision_detail_level = vision_detail_level

    def _get_browser_state_description(self) -> str:
        """Get the browser state description.
        
        Formats the current browser state including URL and interactive
        elements for inclusion in the prompt.
        
        Returns:
            Formatted browser state string.
        """
        elements_text = self.dom_state.element_tree if self.dom_state else 'empty page'
        
        # Truncate if too long
        max_length = 40000
        if len(elements_text) > max_length:
            elements_text = elements_text[:max_length]
            truncated_text = f' (truncated to {max_length} characters)'
        else:
            truncated_text = ''

        browser_state = f"""Current URL: {self.url}

Interactive elements{truncated_text}:
{elements_text}
"""
        return browser_state

    def _get_agent_state_description(self) -> str:
        """Get the agent state description.
        
        Formats the agent's current state including task, step info,
        and current date.
        
        Returns:
            Formatted agent state string.
        """
        step_info_description = ''
        if self.step_info:
            step_info_description = f'Step {self.step_info.step_number + 1}\n'

        time_str = datetime.now().strftime('%Y-%m-%d')
        step_info_description += f'Today: {time_str}'

        agent_state = f"""
<user_request>
{self.task}
</user_request>
<step_info>{step_info_description}</step_info>
"""
        return agent_state

    def get_user_message(self, use_vision: bool = True) -> HumanMessage:
        """Get complete state as a user message.
        
        Builds the final user message combining all state information.
        Optionally includes the screenshot for vision-enabled models.
        
        Args:
            use_vision: Whether to include screenshot in the message.
            
        Returns:
            HumanMessage with text content and optional image.
        """
        # Build state description
        state_description = ''
        
        if self.agent_history_description:
            state_description += f'<agent_history>\n{self.agent_history_description.strip()}\n</agent_history>\n\n'
        
        state_description += f'<agent_state>\n{self._get_agent_state_description().strip()}\n</agent_state>\n'
        state_description += f'<browser_state>\n{self._get_browser_state_description().strip()}\n</browser_state>\n'

        if self.action_descriptions:
            state_description += f'<available_actions>\n{self.action_descriptions}\n</available_actions>\n'

        if use_vision and self.screenshot:
            content = [
                {"type": "text", "text": state_description},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{self.screenshot}",
                        "detail": self.vision_detail_level if self.vision_detail_level != 'auto' else 'auto',
                    },
                },
            ]
            return HumanMessage(content=content)

        return HumanMessage(content=state_description)

