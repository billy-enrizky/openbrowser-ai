"""Shared lightweight models used across openbrowser modules."""

from __future__ import annotations

from pydantic import BaseModel, model_validator


class ActionResult(BaseModel):
	"""Result of executing an action"""

	# For done action
	is_done: bool | None = False
	success: bool | None = None

	# Error handling - always include in long term memory
	error: str | None = None

	# Files
	attachments: list[str] | None = None  # Files to display in the done message

	# Always include in long term memory
	long_term_memory: str | None = None  # Memory of this action

	# if update_only_read_state is True we add the extracted_content to the agent context only once for the next step
	# if update_only_read_state is False we add the extracted_content to the agent long term memory if no long_term_memory is provided
	extracted_content: str | None = None
	include_extracted_content_only_once: bool = False  # Whether the extracted content should be used to update the read_state

	# Metadata for observability (e.g., click coordinates)
	metadata: dict | None = None

	# Deprecated
	include_in_memory: bool = False  # whether to include in extracted_content inside long_term_memory

	@model_validator(mode='after')
	def validate_success_requires_done(self):
		"""Ensure success=True can only be set when is_done=True"""
		if self.success is True and self.is_done is not True:
			raise ValueError(
				'success=True can only be set when is_done=True. '
				'For regular actions that succeed, leave success as None. '
				'Use success=False only for actions that fail.'
			)
		return self
