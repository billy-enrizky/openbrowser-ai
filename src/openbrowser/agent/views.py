"""Agent views and Pydantic models following browser-use pattern."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator, model_validator
from uuid_extensions import uuid7str

if TYPE_CHECKING:
    from openbrowser.llm.base import BaseChatModel
    from openbrowser.agent.message_manager.views import MessageManagerState
    from openbrowser.filesystem.views import FileSystemState

logger = logging.getLogger(__name__)

# Default include attributes for DOM serialization
DEFAULT_INCLUDE_ATTRIBUTES = [
    'title', 'type', 'name', 'role', 'aria-label', 'aria-expanded',
    'aria-haspopup', 'aria-selected', 'aria-checked', 'placeholder',
    'value', 'alt', 'href', 'src', 'data-testid', 'data-id',
]


class AgentSettings(BaseModel):
    """Configuration options for the Agent.
    
    Defines behavioral settings and limits for the browser automation agent.
    All settings have sensible defaults that work for most use cases.
    
    Attributes:
        use_vision: Enable screenshot analysis. 'auto' enables for supported models.
        vision_detail_level: Image detail level ('auto', 'low', 'high').
        save_conversation_path: Path to save conversation history.
        save_conversation_path_encoding: Encoding for saved conversations.
        max_failures: Maximum consecutive failures before stopping.
        max_actions_per_step: Maximum actions per LLM step.
        use_thinking: Enable thinking/reasoning in agent output.
        flash_mode: Enable fast mode with reduced output fields.
        max_history_items: Limit history items in context (None for unlimited).
        step_timeout: Timeout per step in seconds.
        llm_timeout: Timeout for LLM calls in seconds.
        generate_gif: Generate GIF from screenshots (False, True, or path).
        include_attributes: DOM attributes to include in serialization.
        calculate_cost: Track and calculate token costs.
        
    Example:
        >>> settings = AgentSettings(
        ...     use_vision=True,
        ...     max_actions_per_step=4,
        ...     max_failures=3
        ... )
    """

    use_vision: bool | Literal['auto'] = 'auto'
    vision_detail_level: Literal['auto', 'low', 'high'] = 'auto'
    save_conversation_path: str | Path | None = None
    save_conversation_path_encoding: str | None = 'utf-8'
    max_failures: int = 3
    max_actions_per_step: int = 10
    use_thinking: bool = True
    flash_mode: bool = False
    max_history_items: int | None = None
    step_timeout: int = 180
    llm_timeout: int = 60
    # Additional settings from browser-use
    generate_gif: bool | str = False
    override_system_message: str | None = None
    extend_system_message: str | None = None
    include_attributes: list[str] | None = DEFAULT_INCLUDE_ATTRIBUTES
    page_extraction_llm: Any | None = None  # BaseChatModel but avoiding import cycle
    calculate_cost: bool = False
    include_tool_call_examples: bool = False
    final_response_after_failure: bool = True


class AgentState(BaseModel):
    """Holds all state information for an Agent - serializable for checkpointing.
    
    Contains the complete state needed to pause, resume, or checkpoint
    an agent's execution. All fields are designed to be JSON-serializable.
    
    Attributes:
        agent_id: Unique identifier for this agent instance.
        n_steps: Current step number.
        consecutive_failures: Count of consecutive failed steps.
        last_result: Results from the most recent action execution.
        last_plan: The agent's last planning output.
        last_model_output: Complete output from last LLM call.
        paused: Whether the agent is currently paused.
        stopped: Whether the agent has been stopped.
        session_initialized: Whether browser session is ready.
        follow_up_task: Whether this is a follow-up to a previous task.
        message_manager_state: Serialized message manager state.
        file_system_state: Serialized file system state.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent_id: str = Field(default_factory=uuid7str)
    n_steps: int = 1
    consecutive_failures: int = 0
    last_result: list[Any] | None = None  # list[ActionResult]
    last_plan: str | None = None
    last_model_output: Any | None = None  # AgentOutput

    # Pause/resume state
    paused: bool = False
    stopped: bool = False
    session_initialized: bool = False
    follow_up_task: bool = False

    # State containers (lazy imported types)
    message_manager_state: Any | None = None  # MessageManagerState
    file_system_state: Any | None = None  # FileSystemState


class ActionResult(BaseModel):
    """Result of executing an action.
    
    Contains the outcome of a single action execution including success/failure
    status, any extracted content, and error information.
    
    Attributes:
        is_done: True if this action completes the task.
        success: True if task completed successfully (only valid when is_done=True).
        error: Error message if the action failed.
        extracted_content: Content extracted from the page.
        long_term_memory: Information to remember across steps.
        include_extracted_content_only_once: Don't repeat content in history.
        attachments: List of file paths or URLs attached to result.
        metadata: Additional metadata about the action.
        
    Note:
        success=True can only be set when is_done=True. For regular
        successful actions, leave success as None.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    is_done: bool | None = False
    success: bool | None = None
    error: str | None = None
    extracted_content: str | None = None
    long_term_memory: str | None = None
    include_extracted_content_only_once: bool = False
    attachments: list[str] | None = None
    metadata: dict | None = None

    @model_validator(mode='after')
    def validate_success_requires_done(self):
        """Ensure success=True can only be set when is_done=True."""
        if self.success is True and self.is_done is not True:
            raise ValueError(
                'success=True can only be set when is_done=True. '
                'For regular actions that succeed, leave success as None.'
            )
        return self


@dataclass
class AgentStepInfo:
    """Information about the current step.
    
    Provides context about the agent's progress through its execution.
    
    Attributes:
        step_number: Current step number (0-indexed).
        max_steps: Maximum steps allowed.
    """
    
    step_number: int
    max_steps: int

    def is_last_step(self) -> bool:
        """Check if this is the last step.
        
        Returns:
            True if step_number >= max_steps - 1.
        """
        return self.step_number >= self.max_steps - 1


class StepMetadata(BaseModel):
    """Metadata for a single step including timing.
    
    Records timing information for each step to enable performance
    analysis and debugging.
    
    Attributes:
        step_start_time: Unix timestamp when step started.
        step_end_time: Unix timestamp when step completed.
        step_number: The step number this metadata describes.
    """

    step_start_time: float
    step_end_time: float
    step_number: int

    @property
    def duration_seconds(self) -> float:
        """Calculate step duration in seconds.
        
        Returns:
            Duration of the step in seconds.
        """
        return self.step_end_time - self.step_start_time


class AgentBrain(BaseModel):
    """Agent's current mental state.
    
    Represents the agent's reasoning and planning state at a given point.
    Used to track the agent's thought process across steps.
    
    Attributes:
        thinking: The agent's reasoning about the current situation.
        evaluation_previous_goal: Assessment of last action (Success/Failure).
        memory: Important context the agent is remembering.
        next_goal: The agent's immediate next objective.
    """
    
    thinking: str | None = None
    evaluation_previous_goal: str
    memory: str
    next_goal: str


class AgentOutput(BaseModel):
    """Single unified output from LLM per step.
    
    This model combines all agent reasoning with actions in a single response.
    It represents the complete output from the LLM for one agent step.
    
    Attributes:
        thinking: Agent's reasoning about current situation.
        evaluation_previous_goal: Assessment of previous action's success.
        memory: Important context to remember.
        next_goal: Immediate next objective.
        action: List of actions to execute (minimum 1).
        
    Example:
        >>> output = AgentOutput(
        ...     evaluation_previous_goal="Success: Page loaded",
        ...     memory="Looking for login button",
        ...     next_goal="Click the login button",
        ...     action=[ClickAction(index=5)]
        ... )
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    thinking: str | None = None
    evaluation_previous_goal: str | None = None
    memory: str | None = None
    next_goal: str | None = None
    action: list[Any] = Field(
        ...,
        json_schema_extra={'min_items': 1},
    )

    @classmethod
    def model_json_schema(cls, **kwargs):
        schema = super().model_json_schema(**kwargs)
        schema['required'] = ['evaluation_previous_goal', 'memory', 'next_goal', 'action']
        return schema

    @property
    def current_state(self) -> AgentBrain:
        """Return an AgentBrain with the current state properties.
        
        Extracts the reasoning fields from this output into an AgentBrain
        instance for easier access.
        
        Returns:
            AgentBrain instance with thinking, evaluation, memory, and goal.
        """
        return AgentBrain(
            thinking=self.thinking,
            evaluation_previous_goal=self.evaluation_previous_goal or '',
            memory=self.memory or '',
            next_goal=self.next_goal or '',
        )

    @staticmethod
    def type_with_custom_actions(custom_actions: type) -> type[AgentOutput]:
        """Create AgentOutput with dynamic action types from registry.
        
        Generates a new AgentOutput subclass with the action field typed
        to accept the provided custom action types.
        
        Args:
            custom_actions: Union type of allowed action classes.
            
        Returns:
            New AgentOutput subclass with custom action types.
        """
        model_ = create_model(
            'AgentOutput',
            __base__=AgentOutput,
            action=(
                list[custom_actions],
                Field(..., description='List of actions to execute', json_schema_extra={'min_items': 1}),
            ),
            __module__=AgentOutput.__module__,
        )
        return model_

    @staticmethod
    def type_with_custom_actions_flash_mode(custom_actions: type) -> type[AgentOutput]:
        """Create AgentOutput for flash mode - memory and action fields only.
        
        Flash mode reduces token usage by omitting thinking, evaluation,
        and next_goal fields from the output schema.
        
        Args:
            custom_actions: Union type of allowed action classes.
            
        Returns:
            New AgentOutput subclass for flash mode operation.
        """

        class AgentOutputFlashMode(AgentOutput):
            @classmethod
            def model_json_schema(cls, **kwargs):
                schema = super().model_json_schema(**kwargs)
                del schema['properties']['thinking']
                del schema['properties']['evaluation_previous_goal']
                del schema['properties']['next_goal']
                schema['required'] = ['memory', 'action']
                return schema

        model = create_model(
            'AgentOutput',
            __base__=AgentOutputFlashMode,
            action=(
                list[custom_actions],
                Field(..., json_schema_extra={'min_items': 1}),
            ),
            __module__=AgentOutputFlashMode.__module__,
        )
        return model


class BrowserStateHistory(BaseModel):
    """Browser state snapshot for history.
    
    Captures the browser state at a specific point in time for
    inclusion in the agent's execution history.
    
    Attributes:
        url: Current page URL.
        title: Current page title.
        screenshot: Base64-encoded screenshot data.
        screenshot_path: Path to saved screenshot file.
        interacted_element: Elements that were interacted with.
    """
    
    url: str | None = None
    title: str | None = None
    screenshot: str | None = None
    screenshot_path: str | None = None
    interacted_element: list[dict | None] | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary.
        
        Returns:
            Dict representation with None values excluded.
        """
        return self.model_dump(exclude_none=True)

    def get_screenshot(self) -> str | None:
        """Get screenshot as base64 string.
        
        Returns the screenshot data, either from the stored base64 string
        or by reading from the screenshot_path file.
        
        Returns:
            Base64-encoded screenshot string, or None if not available.
        """
        if self.screenshot:
            return self.screenshot
        if self.screenshot_path:
            try:
                import base64
                with open(self.screenshot_path, 'rb') as f:
                    return base64.b64encode(f.read()).decode('utf-8')
            except Exception:
                pass
        return None


class AgentHistory(BaseModel):
    """History item for agent actions.
    
    Records a single step in the agent's execution including the LLM
    output, action results, browser state, and timing metadata.
    
    Attributes:
        model_output: The LLM's output for this step.
        result: List of ActionResult from executing the actions.
        state: Browser state snapshot at this step.
        metadata: Timing and step number metadata.
        state_message: Additional state context message.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    model_output: AgentOutput | None
    result: list[ActionResult]
    state: BrowserStateHistory
    metadata: StepMetadata | None = None
    state_message: str | None = None

    @field_validator('result', mode='before')
    @classmethod
    def validate_result_list(cls, v: Any) -> list[ActionResult]:
        """Ensure result list contains ActionResult instances.
        
        Validates and converts items in the result list to ActionResult
        instances, handling both direct instances and dict representations.
        
        Args:
            v: Input value to validate.
            
        Returns:
            List of validated ActionResult instances.
            
        Raises:
            ValueError: If items cannot be converted to ActionResult.
        """
        if not isinstance(v, list):
            raise ValueError('result must be a list')
        validated_results = []
        for item in v:
            # Check if it's an ActionResult by checking class name and module
            # This handles cases where isinstance fails due to import path issues
            item_type = type(item)
            is_action_result = (
                item_type.__name__ == "ActionResult" and
                "openbrowser" in item_type.__module__
            )

            if is_action_result:
                validated_results.append(item)
            elif isinstance(item, dict):
                # Convert dict to ActionResult
                validated_results.append(ActionResult(**item))
            else:
                raise ValueError(
                    f"result items must be ActionResult instances or dicts, got {item_type}"
                )
        return validated_results

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Custom serialization."""
        model_output_dump = None
        if self.model_output:
            action_dump = []
            for action in self.model_output.action:
                if hasattr(action, 'model_dump'):
                    action_dump.append(action.model_dump(exclude_none=True))
                else:
                    action_dump.append(action)

            model_output_dump = {
                'evaluation_previous_goal': self.model_output.evaluation_previous_goal,
                'memory': self.model_output.memory,
                'next_goal': self.model_output.next_goal,
                'action': action_dump,
            }
            if self.model_output.thinking is not None:
                model_output_dump['thinking'] = self.model_output.thinking

        result_dump = [r.model_dump(exclude_none=True) for r in self.result]

        return {
            'model_output': model_output_dump,
            'result': result_dump,
            'state': self.state.to_dict(),
            'metadata': self.metadata.model_dump() if self.metadata else None,
            'state_message': self.state_message,
        }


AgentStructuredOutput = TypeVar('AgentStructuredOutput', bound=BaseModel)


class AgentHistoryList(BaseModel, Generic[AgentStructuredOutput]):
    """List of AgentHistory items - the history of agent actions and thoughts.
    
    Maintains the complete execution history of an agent run, providing
    methods for querying, saving, and loading history data.
    
    Attributes:
        history: List of AgentHistory items in chronological order.
        
    Example:
        >>> history = AgentHistoryList()
        >>> history.add_item(agent_history_item)
        >>> print(history.number_of_steps())
        >>> history.save_to_file('history.json')
    """

    history: list[AgentHistory] = Field(default_factory=list)
    _output_model_schema: type[AgentStructuredOutput] | None = None

    def __len__(self) -> int:
        return len(self.history)

    def add_item(self, history_item: AgentHistory) -> None:
        """Add a history item to the list.
        
        Args:
            history_item: The AgentHistory item to append.
        """
        self.history.append(history_item)

    def total_duration_seconds(self) -> float:
        """Get total duration of all steps in seconds.
        
        Sums up the duration of all steps that have metadata.
        
        Returns:
            Total execution time in seconds.
        """
        total = 0.0
        for h in self.history:
            if h.metadata:
                total += h.metadata.duration_seconds
        return total

    def save_to_file(self, filepath: str | Path, sensitive_data: dict | None = None) -> None:
        """Save history to JSON file, optionally filtering sensitive data.
        
        Args:
            filepath: Path to save the JSON file.
            sensitive_data: Dict of values to filter (replaced with '***FILTERED***').
            
        Raises:
            IOError: If the file cannot be written.
        """
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            data = self.model_dump()
            
            # Filter sensitive data if provided
            if sensitive_data:
                data = self._filter_sensitive_data(data, sensitive_data)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            raise e

    def _filter_sensitive_data(self, data: dict, sensitive_data: dict) -> dict:
        """Recursively filter sensitive values from data.
        
        Replaces any occurrence of sensitive values with '***FILTERED***'.
        
        Args:
            data: Data structure to filter.
            sensitive_data: Dict of sensitive values to replace.
            
        Returns:
            Filtered copy of the data.
        """
        import copy
        filtered = copy.deepcopy(data)
        
        sensitive_values = set()
        for key, value in sensitive_data.items():
            if isinstance(value, dict):
                sensitive_values.update(value.values())
            else:
                sensitive_values.add(value)
        
        def filter_value(obj):
            if isinstance(obj, dict):
                return {k: filter_value(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [filter_value(v) for v in obj]
            elif isinstance(obj, str):
                for sv in sensitive_values:
                    if sv in obj:
                        obj = obj.replace(sv, '***FILTERED***')
                return obj
            return obj
        
        return filter_value(filtered)

    @classmethod
    def load_from_file(
        cls,
        filepath: str | Path,
        output_model: type | None = None
    ) -> 'AgentHistoryList':
        """Load history from JSON file.
        
        Deserializes a previously saved history file back into an
        AgentHistoryList instance.
        
        Args:
            filepath: Path to the JSON file.
            output_model: Optional output model type for parsing agent outputs.
                If not provided, uses the base AgentOutput class.
            
        Returns:
            Loaded AgentHistoryList with all history items.
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        history_items = []
        for item_data in data.get('history', []):
            # Parse model_output if output_model is provided
            model_output = None
            if item_data.get('model_output') and output_model:
                try:
                    model_output = output_model(**item_data['model_output'])
                except Exception:
                    # Fall back to AgentOutput if specific model fails
                    model_output = AgentOutput(**item_data['model_output'])
            elif item_data.get('model_output'):
                model_output = AgentOutput(**item_data['model_output'])
            
            # Parse result
            result = []
            for r in item_data.get('result', []):
                if r:
                    result.append(ActionResult(**r))
            
            # Parse state
            state_data = item_data.get('state', {})
            state = BrowserStateHistory(**state_data) if state_data else BrowserStateHistory()
            
            # Parse metadata
            metadata_data = item_data.get('metadata')
            metadata = StepMetadata(**metadata_data) if metadata_data else None
            
            history_items.append(AgentHistory(
                model_output=model_output,
                result=result,
                state=state,
                metadata=metadata,
            ))
        
        instance = cls(history=history_items)
        if output_model:
            instance._output_model_schema = output_model
        return instance

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Custom serialization."""
        return {
            'history': [h.model_dump(**kwargs) for h in self.history],
        }

    def errors(self) -> list[str | None]:
        """Get all errors from history.
        
        Extracts the first error from each step's results.
        
        Returns:
            List with error strings or None for each step.
        """
        errors = []
        for h in self.history:
            step_errors = [r.error for r in h.result if r.error]
            errors.append(step_errors[0] if step_errors else None)
        return errors

    def final_result(self) -> str | None:
        """Get final result from history.
        
        Returns the extracted_content from the last action result.
        
        Returns:
            The final extracted content, or None if not available.
        """
        if self.history and self.history[-1].result:
            last_result = self.history[-1].result[-1]
            return last_result.extracted_content
        return None

    def is_done(self) -> bool:
        """Check if the agent is done.
        
        Returns:
            True if the last action result has is_done=True.
        """
        if self.history and self.history[-1].result:
            return self.history[-1].result[-1].is_done is True
        return False

    def is_successful(self) -> bool | None:
        """Check if the agent completed successfully.
        
        Returns:
            True if done with success=True, False if done with success=False,
            None if not done or no result available.
        """
        if self.history and self.history[-1].result:
            last_result = self.history[-1].result[-1]
            if last_result.is_done is True:
                return last_result.success
        return None

    def urls(self) -> list[str | None]:
        """Get all URLs from history.
        
        Returns:
            List of URLs visited at each step.
        """
        return [h.state.url for h in self.history]

    def screenshots(self, n_last: int | None = None) -> list[str | None]:
        """Get screenshots from history as base64 strings.
        
        Args:
            n_last: Number of most recent screenshots to return.
                None returns all screenshots.
                
        Returns:
            List of base64-encoded screenshot strings.
        """
        if n_last == 0:
            return []
        history_items = self.history if n_last is None else self.history[-n_last:]
        return [item.state.get_screenshot() for item in history_items]

    def model_outputs(self) -> list[AgentOutput]:
        """Get all model outputs from history.
        
        Returns:
            List of AgentOutput instances from each step.
        """
        return [h.model_output for h in self.history if h.model_output]

    def action_results(self) -> list[ActionResult]:
        """Get all results from history.
        
        Flattens all ActionResult instances from all steps.
        
        Returns:
            List of all ActionResult instances.
        """
        results = []
        for h in self.history:
            results.extend([r for r in h.result if r])
        return results

    def number_of_steps(self) -> int:
        """Get the number of steps in the history.
        
        Returns:
            Count of history items.
        """
        return len(self.history)


class AgentError:
    """Container for agent error handling.
    
    Provides error message constants and formatting utilities for
    consistent error handling across the agent.
    
    Attributes:
        VALIDATION_ERROR: Message for invalid model output format.
        RATE_LIMIT_ERROR: Message for rate limiting.
        NO_VALID_ACTION: Message when no action can be parsed.
    """

    VALIDATION_ERROR = 'Invalid model output format. Please follow the correct schema.'
    RATE_LIMIT_ERROR = 'Rate limit reached. Waiting before retry.'
    NO_VALID_ACTION = 'No valid action found'

    @staticmethod
    def format_error(error: Exception, include_trace: bool = False) -> str:
        """Format error message based on error type.
        
        Creates user-friendly error messages with optional stack traces.
        Handles ValidationError specially for better debugging.
        
        Args:
            error: The exception to format.
            include_trace: Whether to include the full stack trace.
            
        Returns:
            Formatted error message string.
        """
        import traceback
        from pydantic import ValidationError

        if isinstance(error, ValidationError):
            return f'{AgentError.VALIDATION_ERROR}\nDetails: {str(error)}'

        error_str = str(error)
        if 'LLM response missing required fields' in error_str:
            lines = error_str.split('\n')
            main_error = lines[0] if lines else error_str
            helpful_msg = f'{main_error}\n\nPlease stick to the required output format.\n'
            if include_trace:
                helpful_msg += f'\n\nFull stacktrace:\n{traceback.format_exc()}'
            return helpful_msg

        if include_trace:
            return f'{str(error)}\nStacktrace:\n{traceback.format_exc()}'
        return str(error)


# --------------------------------------------------------------------------
# Message types for LLM serialization
# --------------------------------------------------------------------------


class ImageURLDetail(BaseModel):
    """Image URL with detail level specification.
    
    Used in vision-enabled messages to specify image content and
    processing detail level.
    
    Attributes:
        url: The image URL (can be data: URL for base64 images).
        detail: Processing detail level ('auto', 'low', 'high').
    """
    
    url: str
    detail: Literal['auto', 'low', 'high'] = 'auto'


class ContentPartTextParam(BaseModel):
    """Text content part for messages.
    
    Represents a text segment in a multi-part message content array.
    
    Attributes:
        type: Always 'text'.
        text: The text content.
    """
    
    type: Literal['text'] = 'text'
    text: str


class ContentPartImageParam(BaseModel):
    """Image content part for messages.
    
    Represents an image in a multi-part message content array.
    
    Attributes:
        type: Always 'image_url'.
        image_url: ImageURLDetail with URL and detail level.
    """
    
    type: Literal['image_url'] = 'image_url'
    image_url: ImageURLDetail


class ContentPartRefusalParam(BaseModel):
    """Refusal content part for assistant messages.
    
    Represents a refusal response from the model.
    
    Attributes:
        type: Always 'refusal'.
        refusal: The refusal message text.
    """
    
    type: Literal['refusal'] = 'refusal'
    refusal: str


class FunctionCall(BaseModel):
    """Function call information.
    
    Contains the name and arguments for a function/tool call.
    
    Attributes:
        name: The function name.
        arguments: JSON string of function arguments.
    """
    
    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    """Tool call from assistant message.
    
    Represents a tool invocation request from the model.
    
    Attributes:
        id: Unique identifier for this tool call.
        type: Always 'function'.
        function: The function call details.
    """
    
    id: str
    type: Literal['function'] = 'function'
    function: FunctionCall


class BaseMessage(BaseModel):
    """Base class for all message types.
    
    Provides common configuration for message serialization.
    All message types inherit from this class.
    """
    
    model_config = ConfigDict(extra='allow')


class SystemMessage(BaseMessage):
    """System message to LLM.
    
    Contains instructions and context for the model.
    
    Attributes:
        role: Always 'system'.
        content: The system prompt text or content parts.
        name: Optional name for the message author.
    """
    
    role: Literal['system'] = 'system'
    content: str | list[ContentPartTextParam]
    name: str | None = None


class UserMessage(BaseMessage):
    """User message to LLM.
    
    Contains user input, which may include text and images.
    
    Attributes:
        role: Always 'user'.
        content: Text string or list of text/image content parts.
        name: Optional name for the user.
    """
    
    role: Literal['user'] = 'user'
    content: str | list[ContentPartTextParam | ContentPartImageParam]
    name: str | None = None


class AssistantMessage(BaseMessage):
    """Assistant message from LLM.
    
    Contains the model's response, which may include text and tool calls.
    
    Attributes:
        role: Always 'assistant'.
        content: Text response or content parts, may be None if tool_calls present.
        tool_calls: List of tool/function calls requested by the model.
        refusal: Refusal message if the model declined to respond.
        name: Optional name for the assistant.
    """
    
    role: Literal['assistant'] = 'assistant'
    content: str | list[ContentPartTextParam | ContentPartRefusalParam] | None = None
    tool_calls: list[ToolCall] | None = None
    refusal: str | None = None
    name: str | None = None


class ToolMessage(BaseMessage):
    """Tool response message.
    
    Contains the result of a tool/function execution.
    
    Attributes:
        role: Always 'tool'.
        content: The tool's response content.
        tool_call_id: ID linking this response to the original tool call.
        name: Optional tool name.
    """
    
    role: Literal['tool'] = 'tool'
    content: str
    tool_call_id: str
    name: str | None = None

