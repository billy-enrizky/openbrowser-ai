"""Registry service for action registration and execution."""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Callable
from inspect import Parameter, iscoroutinefunction, signature
from typing import Any, Generic, Optional, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel, Field, RootModel, create_model

from src.openbrowser.tools.registry.views import (
    ActionModel,
    ActionRegistry,
    RegisteredAction,
)

Context = TypeVar('Context')

logger = logging.getLogger(__name__)


class Registry(Generic[Context]):
    """Service for registering and managing actions."""

    def __init__(self, exclude_actions: list[str] | None = None):
        self.registry = ActionRegistry()
        self.exclude_actions = exclude_actions if exclude_actions is not None else []

    def _get_special_param_types(self) -> dict[str, type | None]:
        """Get the expected types for special parameters."""
        return {
            'context': None,
            'browser_session': None,  # BrowserSession
            'page_url': str,
            'cdp_client': None,
        }

    def _create_param_model(self, function: Callable) -> type[BaseModel]:
        """Create a Pydantic model from function signature."""
        sig = signature(function)
        special_param_names = set(self._get_special_param_types().keys())
        params = {
            name: (param.annotation if param.annotation != Parameter.empty else str, 
                   ... if param.default == param.empty else param.default)
            for name, param in sig.parameters.items()
            if name not in special_param_names
        }
        return create_model(
            f'{function.__name__}_parameters',
            __base__=ActionModel,
            **params,
        )

    def _normalize_action_function_signature(
        self,
        func: Callable,
        description: str,
        param_model: type[BaseModel] | None = None,
    ) -> tuple[Callable, type[BaseModel]]:
        """Normalize action function to accept kwargs with params object."""
        sig = signature(func)
        parameters = list(sig.parameters.values())
        special_param_types = self._get_special_param_types()
        special_param_names = set(special_param_types.keys())

        # Separate special and action parameters
        action_params = []
        special_params = []
        param_model_provided = param_model is not None

        for i, param in enumerate(parameters):
            if i == 0 and param_model_provided and param.name not in special_param_names:
                continue
            if param.name in special_param_names:
                special_params.append(param)
            else:
                action_params.append(param)

        # Create or use param model
        if not param_model_provided:
            if action_params:
                params_dict = {}
                for param in action_params:
                    annotation = param.annotation if param.annotation != Parameter.empty else str
                    default = ... if param.default == Parameter.empty else param.default
                    params_dict[param.name] = (annotation, default)
                param_model = create_model(
                    f'{func.__name__}_Params', 
                    __base__=ActionModel, 
                    **params_dict
                )
            else:
                param_model = create_model(f'{func.__name__}_Params', __base__=ActionModel)

        assert param_model is not None

        @functools.wraps(func)
        async def normalized_wrapper(*args, params: BaseModel | None = None, **kwargs):
            """Normalized action that accepts kwargs."""
            if args:
                raise TypeError(f'{func.__name__}() does not accept positional arguments')

            call_args = []
            params_dict = params.model_dump() if params is not None else {}

            for i, param in enumerate(parameters):
                if param_model_provided and i == 0 and param.name not in special_param_names:
                    call_args.append(params)
                elif param.name in special_param_names:
                    if param.name in kwargs:
                        call_args.append(kwargs[param.name])
                    elif param.default != Parameter.empty:
                        call_args.append(param.default)
                    else:
                        call_args.append(None)
                else:
                    if param.name in params_dict:
                        call_args.append(params_dict[param.name])
                    elif param.default != Parameter.empty:
                        call_args.append(param.default)
                    else:
                        raise ValueError(f"{func.__name__}() missing required parameter '{param.name}'")

            if iscoroutinefunction(func):
                return await func(*call_args)
            else:
                return await asyncio.to_thread(func, *call_args)

        return normalized_wrapper, param_model

    def action(
        self,
        description: str,
        param_model: type[BaseModel] | None = None,
        domains: list[str] | None = None,
    ):
        """Decorator for registering actions."""
        def decorator(func: Callable):
            if func.__name__ in self.exclude_actions:
                return func

            normalized_func, actual_param_model = self._normalize_action_function_signature(
                func, description, param_model
            )

            action = RegisteredAction(
                name=func.__name__,
                description=description,
                function=normalized_func,
                param_model=actual_param_model,
                domains=domains,
            )
            self.registry.actions[func.__name__] = action
            return normalized_func

        return decorator

    async def execute_action(
        self,
        action_name: str,
        params: dict,
        browser_session: Any = None,
        page_url: str | None = None,
        cdp_client: Any = None,
        context: Any = None,
    ) -> Any:
        """Execute a registered action."""
        if action_name not in self.registry.actions:
            raise ValueError(f'Action {action_name} not found')

        action = self.registry.actions[action_name]
        try:
            validated_params = action.param_model(**params)
            special_context = {
                'browser_session': browser_session,
                'page_url': page_url,
                'cdp_client': cdp_client,
                'context': context,
            }
            return await action.function(params=validated_params, **special_context)
        except Exception as e:
            raise RuntimeError(f'Error executing action {action_name}: {str(e)}') from e

    def create_action_model(
        self, 
        include_actions: list[str] | None = None, 
        page_url: str | None = None
    ) -> type[ActionModel]:
        """Create a Union of action models for structured output.
        
        This dynamically creates a model from all registered actions that the LLM
        can use with structured output.
        """
        available_actions: dict[str, RegisteredAction] = {}
        
        for name, action in self.registry.actions.items():
            if include_actions is not None and name not in include_actions:
                continue

            if page_url is None:
                if action.domains is None:
                    available_actions[name] = action
                continue

            if self.registry._match_domains(action.domains, page_url):
                available_actions[name] = action

        # Create individual action models for each action
        individual_action_models: list[type[BaseModel]] = []

        for name, action in available_actions.items():
            individual_model = create_model(
                f'{name.title().replace("_", "")}ActionModel',
                __base__=ActionModel,
                **{
                    name: (
                        action.param_model,
                        Field(description=action.description),
                    )
                },
            )
            individual_action_models.append(individual_model)

        if not individual_action_models:
            return create_model('EmptyActionModel', __base__=ActionModel)

        if len(individual_action_models) == 1:
            return individual_action_models[0]

        # Create Union type with RootModel
        union_type = Union[tuple(individual_action_models)]  # type: ignore

        class ActionModelUnion(RootModel[union_type]):  # type: ignore
            def get_index(self) -> int | None:
                if hasattr(self.root, 'get_index'):
                    return self.root.get_index()
                return None

            def set_index(self, index: int):
                if hasattr(self.root, 'set_index'):
                    self.root.set_index(index)

            def model_dump(self, **kwargs):
                if hasattr(self.root, 'model_dump'):
                    return self.root.model_dump(**kwargs)
                return super().model_dump(**kwargs)

        ActionModelUnion.__name__ = 'ActionModel'
        ActionModelUnion.__qualname__ = 'ActionModel'

        return ActionModelUnion  # type: ignore

    def get_prompt_description(self, page_url: str | None = None) -> str:
        """Get a description of all actions for the prompt."""
        return self.registry.get_prompt_description(page_url=page_url)

