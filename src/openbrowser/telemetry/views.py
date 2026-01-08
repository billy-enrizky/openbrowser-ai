"""Telemetry views and event definitions.

Performance optimizations:
- Using dataclasses with slots=True for all event classes
- Cached is_running_in_docker check
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Literal

from openbrowser.config import is_running_in_docker


# Cache the docker check at module level for repeated calls
@lru_cache(maxsize=1)
def _cached_is_docker() -> bool:
	"""Cached version of is_running_in_docker check."""
	return is_running_in_docker()


@dataclass
class BaseTelemetryEvent(ABC):
	@property
	@abstractmethod
	def name(self) -> str:
		pass

	@property
	def properties(self) -> dict[str, Any]:
		props = {k: v for k, v in asdict(self).items() if k != 'name'}
		# Use cached docker check
		props['is_docker'] = _cached_is_docker()
		return props


@dataclass(slots=True)
class AgentTelemetryEvent(BaseTelemetryEvent):
	# start details
	task: str
	model: str
	model_provider: str
	max_steps: int
	max_actions_per_step: int
	use_vision: bool | Literal['auto']
	version: str
	source: str
	cdp_url: str | None
	agent_type: str | None  # 'code' for CodeAgent, None for regular Agent
	# step details
	action_errors: Sequence[str | None]
	action_history: Sequence[list[dict] | None]
	urls_visited: Sequence[str | None]
	# end details
	steps: int
	total_input_tokens: int
	total_output_tokens: int
	prompt_cached_tokens: int
	total_tokens: int
	total_duration_seconds: float
	success: bool | None
	final_result_response: str | None
	error_message: str | None

	name: str = field(default='agent_event', repr=False)

	@property
	def properties(self) -> dict[str, Any]:
		# Override to use cached docker check
		props = {k: v for k, v in asdict(self).items() if k != 'name'}
		props['is_docker'] = _cached_is_docker()
		return props


@dataclass(slots=True)
class MCPClientTelemetryEvent(BaseTelemetryEvent):
	"""Telemetry event for MCP client usage"""

	server_name: str
	command: str
	tools_discovered: int
	version: str
	action: str  # 'connect', 'disconnect', 'tool_call'
	tool_name: str | None = None
	duration_seconds: float | None = None
	error_message: str | None = None

	name: str = field(default='mcp_client_event', repr=False)

	@property
	def properties(self) -> dict[str, Any]:
		props = {k: v for k, v in asdict(self).items() if k != 'name'}
		props['is_docker'] = _cached_is_docker()
		return props


@dataclass(slots=True)
class MCPServerTelemetryEvent(BaseTelemetryEvent):
	"""Telemetry event for MCP server usage"""

	version: str
	action: str  # 'start', 'stop', 'tool_call'
	tool_name: str | None = None
	duration_seconds: float | None = None
	error_message: str | None = None
	parent_process_cmdline: str | None = None

	name: str = field(default='mcp_server_event', repr=False)

	@property
	def properties(self) -> dict[str, Any]:
		props = {k: v for k, v in asdict(self).items() if k != 'name'}
		props['is_docker'] = _cached_is_docker()
		return props


@dataclass(slots=True)
class CLITelemetryEvent(BaseTelemetryEvent):
	"""Telemetry event for CLI usage"""

	version: str
	action: str  # 'start', 'message_sent', 'task_completed', 'error'
	mode: str  # 'interactive', 'oneshot', 'mcp_server'
	model: str | None = None
	model_provider: str | None = None
	duration_seconds: float | None = None
	error_message: str | None = None

	name: str = field(default='cli_event', repr=False)

	@property
	def properties(self) -> dict[str, Any]:
		props = {k: v for k, v in asdict(self).items() if k != 'name'}
		props['is_docker'] = _cached_is_docker()
		return props
