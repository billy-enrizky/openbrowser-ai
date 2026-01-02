"""Views and models for the tools registry.

This module provides the Pydantic models and data classes used by the
action registry system. These models define the structure of registered
actions and their parameters for browser automation.

Classes:
    ActionModel: Base model for action parameter validation.
    RegisteredAction: Metadata and function for a registered action.
    ActionRegistry: Container for all registered actions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ActionModel(BaseModel):
    """Base model for dynamically created action parameter models.
    
    This class serves as the base for all action parameter models. When the
    registry creates parameter models from function signatures, they inherit
    from this class to gain common functionality.
    
    The model is used with Pydantic's create_model() to dynamically generate
    typed parameter models for each registered action.
    
    Attributes:
        model_config: Pydantic configuration allowing arbitrary types and
            forbidding extra fields for strict validation.
    
    Example:
        >>> class ClickParams(ActionModel):
        ...     index: int = Field(description="Element index to click")
        ...     button: str = Field(default="left", description="Mouse button")
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    def get_index(self) -> int | None:
        """Get the index from the action parameters if present.
        
        This utility method searches the action parameters for an 'index'
        field, which is commonly used to reference DOM elements.
        
        Returns:
            int | None: The index value if found, None otherwise.
        """
        params = self.model_dump(exclude_unset=True).values()
        if not params:
            return None
        for param in params:
            if param is not None and isinstance(param, dict) and 'index' in param:
                return param['index']
        return None

    def set_index(self, index: int) -> None:
        """Set the index on the action parameters.
        
        This utility method updates the 'index' field in the action parameters,
        used for remapping element indices after DOM changes.
        
        Args:
            index: The new index value to set.
        """
        action_data = self.model_dump(exclude_unset=True)
        if not action_data:
            return
        action_name = next(iter(action_data.keys()))
        action_params = getattr(self, action_name, None)
        if action_params and hasattr(action_params, 'index'):
            action_params.index = index


class RegisteredAction(BaseModel):
    """Metadata and function reference for a registered action.
    
    This model holds all information about a registered action, including
    its name, description, the function to execute, parameter model,
    and optional domain restrictions.
    
    Attributes:
        name: The action name (e.g., 'click', 'navigate').
        description: Human-readable description for the LLM.
        function: The async callable to execute the action.
        param_model: Pydantic model class for parameter validation.
        domains: Optional list of domain patterns for URL filtering.
    
    Example:
        >>> action = RegisteredAction(
        ...     name="navigate",
        ...     description="Navigate to a URL",
        ...     function=navigate_func,
        ...     param_model=NavigateParams,
        ...     domains=None  # Available on all domains
        ... )
    """

    name: str
    description: str
    function: Callable
    param_model: type[BaseModel]
    domains: list[str] | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def prompt_description(self) -> str:
        """Get a description of the action for the system prompt.
        
        Generates a formatted string that describes the action and its
        parameters for inclusion in the LLM system prompt.
        
        Returns:
            str: Formatted description like "click: Click on element. (index=int)"
        """
        schema = self.param_model.model_json_schema()
        params = []

        if 'properties' in schema:
            for param_name, param_info in schema['properties'].items():
                param_desc = param_name
                if 'type' in param_info:
                    param_desc += f'={param_info["type"]}'
                if 'description' in param_info:
                    param_desc += f' ({param_info["description"]})'
                params.append(param_desc)

        if params:
            return f'{self.name}: {self.description}. ({", ".join(params)})'
        return f'{self.name}: {self.description}'


class ActionRegistry(BaseModel):
    """Container for all registered actions.
    
    This model holds the collection of registered actions and provides
    methods for generating action descriptions for LLM prompts.
    
    Attributes:
        actions: Dictionary mapping action names to RegisteredAction objects.
    
    Example:
        >>> registry = ActionRegistry()
        >>> registry.actions['click'] = click_action
        >>> description = registry.get_prompt_description(page_url="https://google.com")
    """

    actions: dict[str, RegisteredAction] = Field(default_factory=dict)

    def get_prompt_description(self, page_url: str | None = None) -> str:
        """Get a description of all actions for the LLM prompt.
        
        Generates a formatted string describing available actions. When a
        page_url is provided, includes domain-filtered actions for that URL.
        
        Args:
            page_url: Optional current page URL. If provided, includes
                domain-specific actions that match the URL.
            
        Returns:
            str: Newline-separated action descriptions for the prompt.
        """
        if page_url is None:
            # For system prompt, include only actions with no domain filters
            return '\n'.join(
                action.prompt_description() 
                for action in self.actions.values() 
                if action.domains is None
            )

        # Include filtered actions for the current page URL
        filtered_actions = []
        for action in self.actions.values():
            if not action.domains:
                continue
            if self._match_domains(action.domains, page_url):
                filtered_actions.append(action)

        return '\n'.join(action.prompt_description() for action in filtered_actions)

    @staticmethod
    def _match_domains(domains: list[str] | None, url: str) -> bool:
        """Match domain patterns against a URL."""
        if domains is None or not url:
            return True

        from urllib.parse import urlparse
        import fnmatch

        try:
            parsed = urlparse(url)
            hostname = parsed.hostname or ''
        except Exception:
            return False

        for pattern in domains:
            # Handle scheme patterns like http*://*.example.com
            if '://' in pattern:
                scheme_pattern, domain_pattern = pattern.split('://', 1)
                if not fnmatch.fnmatch(parsed.scheme, scheme_pattern.replace('*', '?*')):
                    continue
                if fnmatch.fnmatch(hostname, domain_pattern):
                    return True
            else:
                if fnmatch.fnmatch(hostname, pattern):
                    return True

        return False

