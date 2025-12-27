"""Views and models for the tools registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ActionModel(BaseModel):
    """Base model for dynamically created action models.
    
    Each action is represented as a field on a dynamically created subclass.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    def get_index(self) -> int | None:
        """Get the index from the action parameters if present."""
        params = self.model_dump(exclude_unset=True).values()
        if not params:
            return None
        for param in params:
            if param is not None and isinstance(param, dict) and 'index' in param:
                return param['index']
        return None

    def set_index(self, index: int) -> None:
        """Set the index on the action parameters."""
        action_data = self.model_dump(exclude_unset=True)
        if not action_data:
            return
        action_name = next(iter(action_data.keys()))
        action_params = getattr(self, action_name, None)
        if action_params and hasattr(action_params, 'index'):
            action_params.index = index


class RegisteredAction(BaseModel):
    """Model for a registered action."""

    name: str
    description: str
    function: Callable
    param_model: type[BaseModel]
    domains: list[str] | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def prompt_description(self) -> str:
        """Get a description of the action for the prompt."""
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
    """Model representing the action registry."""

    actions: dict[str, RegisteredAction] = Field(default_factory=dict)

    def get_prompt_description(self, page_url: str | None = None) -> str:
        """Get a description of all actions for the prompt.
        
        Args:
            page_url: If provided, filter actions by URL using domain filters.
            
        Returns:
            A string description of available actions.
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

