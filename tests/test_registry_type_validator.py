"""Tests for registry type validator handling of Python 3.10+ X | Y union syntax."""

import pytest

from openbrowser.browser import BrowserSession
from openbrowser.filesystem.file_system import FileSystem
from openbrowser.tools.registry.service import Registry


@pytest.fixture
def registry():
	return Registry()


class TestUnionTypeSyntax:
	"""Test that the registry handles both typing.Union and types.UnionType (X | Y) syntax."""

	def test_pipe_union_optional_file_system(self, registry: Registry):
		"""FileSystem | None = None should be accepted as a valid special param type."""

		@registry.action('Test action with pipe union file_system')
		async def action_with_pipe_union(index: int, file_system: FileSystem | None = None):
			pass

		assert 'action_with_pipe_union' in registry.registry.actions

	def test_pipe_union_optional_browser_session(self, registry: Registry):
		"""BrowserSession | None = None should be accepted as a valid special param type."""

		@registry.action('Test action with pipe union browser_session')
		async def action_with_pipe_union_bs(index: int, browser_session: BrowserSession | None = None):
			pass

		assert 'action_with_pipe_union_bs' in registry.registry.actions

	def test_pipe_union_required_browser_session(self, registry: Registry):
		"""BrowserSession | None (no default) should be accepted."""

		@registry.action('Test action with required pipe union')
		async def action_required_pipe(index: int, browser_session: BrowserSession | None = None):
			pass

		assert 'action_required_pipe' in registry.registry.actions

	def test_typing_optional_still_works(self, registry: Registry):
		"""typing.Optional[FileSystem] (old syntax) should still work."""
		from typing import Optional

		@registry.action('Test action with Optional')
		async def action_with_optional(index: int, file_system: Optional[FileSystem] = None):
			pass

		assert 'action_with_optional' in registry.registry.actions

	def test_typing_union_still_works(self, registry: Registry):
		"""typing.Union[FileSystem, None] (old syntax) should still work."""
		from typing import Union

		@registry.action('Test action with Union')
		async def action_with_union(index: int, file_system: Union[FileSystem, None] = None):
			pass

		assert 'action_with_union' in registry.registry.actions

	def test_bare_type_still_works(self, registry: Registry):
		"""Bare FileSystem (no Optional/Union) should still work."""

		@registry.action('Test action with bare type')
		async def action_with_bare(index: int, file_system: FileSystem = None):
			pass

		assert 'action_with_bare' in registry.registry.actions

	def test_wrong_type_pipe_union_raises(self, registry: Registry):
		"""str | None for file_system should raise ValueError (wrong type)."""
		with pytest.raises(ValueError, match="conflicts with special argument"):

			@registry.action('Test action with wrong pipe union type')
			async def action_wrong_type(index: int, file_system: str | None = None):
				pass

	def test_pipe_union_available_file_paths(self, registry: Registry):
		"""list[str] | None = None for available_file_paths should be accepted."""

		@registry.action('Test action with pipe union list')
		async def action_with_list_union(index: int, available_file_paths: list[str] | None = None):
			pass

		assert 'action_with_list_union' in registry.registry.actions

	def test_pipe_union_has_sensitive_data(self, registry: Registry):
		"""bool | None for has_sensitive_data should not conflict (bool is expected)."""

		@registry.action('Test action with bool pipe union')
		async def action_with_bool_union(index: int, has_sensitive_data: bool = False):
			pass

		assert 'action_with_bool_union' in registry.registry.actions
