"""Agent views and Pydantic models following browser-use pattern."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator, model_validator

logger = logging.getLogger(__name__)


class AgentSettings(BaseModel):
    """Configuration options for the Agent."""

    use_vision: bool | Literal['auto'] = 'auto'
    vision_detail_level: Literal['auto', 'low', 'high'] = 'auto'
    save_conversation_path: str | Path | None = None
    max_failures: int = 3
    max_actions_per_step: int = 4
    use_thinking: bool = True
    flash_mode: bool = False
    max_history_items: int | None = None
    step_timeout: int = 180


class ActionResult(BaseModel):
    """Result of executing an action."""

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
    """Information about the current step."""
    
    step_number: int
    max_steps: int

    def is_last_step(self) -> bool:
        """Check if this is the last step."""
        return self.step_number >= self.max_steps - 1


class StepMetadata(BaseModel):
    """Metadata for a single step including timing."""

    step_start_time: float
    step_end_time: float
    step_number: int

    @property
    def duration_seconds(self) -> float:
        """Calculate step duration in seconds."""
        return self.step_end_time - self.step_start_time


class AgentBrain(BaseModel):
    """Agent's current mental state."""
    
    thinking: str | None = None
    evaluation_previous_goal: str
    memory: str
    next_goal: str


class AgentOutput(BaseModel):
    """Single unified output from LLM per step.
    
    This model combines all agent reasoning with actions in a single response.
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
        """Return an AgentBrain with the current state properties."""
        return AgentBrain(
            thinking=self.thinking,
            evaluation_previous_goal=self.evaluation_previous_goal or '',
            memory=self.memory or '',
            next_goal=self.next_goal or '',
        )

    @staticmethod
    def type_with_custom_actions(custom_actions: type) -> type[AgentOutput]:
        """Create AgentOutput with dynamic action types from registry."""
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
        """Create AgentOutput for flash mode - memory and action fields only."""

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
    """Browser state snapshot for history."""
    
    url: str | None = None
    title: str | None = None
    screenshot: str | None = None
    screenshot_path: str | None = None
    interacted_element: list[dict | None] | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return self.model_dump(exclude_none=True)

    def get_screenshot(self) -> str | None:
        """Get screenshot as base64 string."""
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
    """History item for agent actions."""

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    model_output: AgentOutput | None
    result: list[ActionResult]
    state: BrowserStateHistory
    metadata: StepMetadata | None = None
    state_message: str | None = None

    @field_validator('result', mode='before')
    @classmethod
    def validate_result_list(cls, v: Any) -> list[ActionResult]:
        """Ensure result list contains ActionResult instances."""
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
    """List of AgentHistory items - the history of agent actions and thoughts."""

    history: list[AgentHistory] = Field(default_factory=list)
    _output_model_schema: type[AgentStructuredOutput] | None = None

    def __len__(self) -> int:
        return len(self.history)

    def add_item(self, history_item: AgentHistory) -> None:
        """Add a history item to the list."""
        self.history.append(history_item)

    def total_duration_seconds(self) -> float:
        """Get total duration of all steps in seconds."""
        total = 0.0
        for h in self.history:
            if h.metadata:
                total += h.metadata.duration_seconds
        return total

    def save_to_file(self, filepath: str | Path, sensitive_data: dict | None = None) -> None:
        """Save history to JSON file, optionally filtering sensitive data."""
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
        """Recursively filter sensitive values from data."""
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
        
        Args:
            filepath: Path to the JSON file
            output_model: Optional output model type for parsing agent outputs
            
        Returns:
            Loaded AgentHistoryList
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
        """Get all errors from history."""
        errors = []
        for h in self.history:
            step_errors = [r.error for r in h.result if r.error]
            errors.append(step_errors[0] if step_errors else None)
        return errors

    def final_result(self) -> str | None:
        """Get final result from history."""
        if self.history and self.history[-1].result:
            last_result = self.history[-1].result[-1]
            return last_result.extracted_content
        return None

    def is_done(self) -> bool:
        """Check if the agent is done."""
        if self.history and self.history[-1].result:
            return self.history[-1].result[-1].is_done is True
        return False

    def is_successful(self) -> bool | None:
        """Check if the agent completed successfully."""
        if self.history and self.history[-1].result:
            last_result = self.history[-1].result[-1]
            if last_result.is_done is True:
                return last_result.success
        return None

    def urls(self) -> list[str | None]:
        """Get all URLs from history."""
        return [h.state.url for h in self.history]

    def screenshots(self, n_last: int | None = None) -> list[str | None]:
        """Get screenshots from history as base64 strings."""
        if n_last == 0:
            return []
        history_items = self.history if n_last is None else self.history[-n_last:]
        return [item.state.get_screenshot() for item in history_items]

    def model_outputs(self) -> list[AgentOutput]:
        """Get all model outputs from history."""
        return [h.model_output for h in self.history if h.model_output]

    def action_results(self) -> list[ActionResult]:
        """Get all results from history."""
        results = []
        for h in self.history:
            results.extend([r for r in h.result if r])
        return results

    def number_of_steps(self) -> int:
        """Get the number of steps in the history."""
        return len(self.history)


class AgentError:
    """Container for agent error handling."""

    VALIDATION_ERROR = 'Invalid model output format. Please follow the correct schema.'
    RATE_LIMIT_ERROR = 'Rate limit reached. Waiting before retry.'
    NO_VALID_ACTION = 'No valid action found'

    @staticmethod
    def format_error(error: Exception, include_trace: bool = False) -> str:
        """Format error message based on error type."""
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
    """Image URL with detail level specification."""
    
    url: str
    detail: Literal['auto', 'low', 'high'] = 'auto'


class ContentPartTextParam(BaseModel):
    """Text content part for messages."""
    
    type: Literal['text'] = 'text'
    text: str


class ContentPartImageParam(BaseModel):
    """Image content part for messages."""
    
    type: Literal['image_url'] = 'image_url'
    image_url: ImageURLDetail


class ContentPartRefusalParam(BaseModel):
    """Refusal content part for assistant messages."""
    
    type: Literal['refusal'] = 'refusal'
    refusal: str


class FunctionCall(BaseModel):
    """Function call information."""
    
    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    """Tool call from assistant message."""
    
    id: str
    type: Literal['function'] = 'function'
    function: FunctionCall


class BaseMessage(BaseModel):
    """Base class for all message types."""
    
    model_config = ConfigDict(extra='allow')


class SystemMessage(BaseMessage):
    """System message to LLM."""
    
    role: Literal['system'] = 'system'
    content: str | list[ContentPartTextParam]
    name: str | None = None


class UserMessage(BaseMessage):
    """User message to LLM."""
    
    role: Literal['user'] = 'user'
    content: str | list[ContentPartTextParam | ContentPartImageParam]
    name: str | None = None


class AssistantMessage(BaseMessage):
    """Assistant message from LLM."""
    
    role: Literal['assistant'] = 'assistant'
    content: str | list[ContentPartTextParam | ContentPartRefusalParam] | None = None
    tool_calls: list[ToolCall] | None = None
    refusal: str | None = None
    name: str | None = None


class ToolMessage(BaseMessage):
    """Tool response message."""
    
    role: Literal['tool'] = 'tool'
    content: str
    tool_call_id: str
    name: str | None = None

